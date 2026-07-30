"""
IO工具模块：文件读写、路径管理

双轨磁盘布局（v2.1）：

    data/features/{elder_id}/features_sleep.csv     8 维 + day_key + quality
    data/features/{elder_id}/features_social.csv    5 维 + day_key + quality

    data/baselines/{elder_id}/gru_sleep.pth         模型（按轨）
                             gru_social.pth
                             scaler_sleep.pkl       归一化基准（按轨）
                             scaler_social.pkl
                             residual_stats_sleep.pkl    残差统计（含 signed+abs 两套）
                             residual_stats_social.pkl
                             ewma_sleep.pkl              EWMA 池
                             ewma_social_weekday.pkl
                             ewma_social_weekend.pkl
                             baseline_meta.json          元信息（两轨共用一个文件）

时间索引用 `day_key`（自然日 D，本地时区）：
    睡眠特征(D) ← 在 D 日早晨结束的那一夜
    日间特征(D) ← [T_rise(D 日早晨), T_bed(D 日晚上)]

为什么不用"入睡时刻所在自然日"做索引：那是循环定义（要先知道日期才能定搜索窗口，
要先有搜索窗口才能找到入睡时刻），且入睡从 23:30 漂到 00:30 时索引键会跳变，
在特征表里制造一个重复日 + 一个空洞日——而就寝相位漂移正是 sleep_onset_clock
要监测的东西，索引键绝不能绑在被监测量上。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline.scaler_utils import get_feature_names, validate_track


# features_{track}.csv 的非特征列
DAY_KEY_COL = "day_key"
META_COLS = ["missing_count", "data_quality"]


# ========== 路径管理 ==========

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


def get_baseline_dir(elder_id: str) -> Path:
    """获取指定老人的基线存储目录"""
    return get_project_root() / "data" / "baselines" / elder_id


def get_features_dir(elder_id: str) -> Path:
    """获取指定老人的特征数据目录"""
    return get_project_root() / "data" / "features" / elder_id


def get_log_dir(log_type: str = "daily_inference") -> Path:
    """获取日志目录"""
    return get_project_root() / "data" / "logs" / log_type


def get_features_path(elder_id: str, track: str) -> Path:
    """某轨特征表的路径：features_sleep.csv / features_social.csv"""
    return get_features_dir(elder_id) / f"features_{validate_track(track)}.csv"


def get_feature_columns(track: str) -> list[str]:
    """某轨特征表的完整列顺序（写 CSV 时必须显式对齐，见 save_daily_features）"""
    return [DAY_KEY_COL] + get_feature_names(track) + META_COLS


# ========== 特征数据读写 ==========

def save_daily_features(
    elder_id: str,
    day_key: str,
    feature_vector: np.ndarray,
    track: str,
    missing_count: int = 0,
    data_quality: str = "valid",
) -> None:
    """
    追加保存某轨的每日特征向量到对应 CSV。

    同一 day_key 重复写入时覆盖旧行（幂等），避免补算历史日产生重复行。

    Args:
        elder_id: 老人ID
        day_key: 自然日 "YYYY-MM-DD"
        feature_vector: (feature_dim,) 该轨特征向量
        track: "sleep" / "social"
        missing_count: 缺失特征计数
        data_quality: valid / degraded / insufficient / offline
    """
    names = get_feature_names(track)
    expected = len(names)
    vec = np.asarray(feature_vector, dtype=np.float64).flatten()
    if vec.shape[0] != expected:
        raise ValueError(
            f"Track {track!r} expects {expected} features, got {vec.shape[0]}"
        )

    dir_path = get_features_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = get_features_path(elder_id, track)

    row: dict[str, Any] = {name: float(vec[i]) for i, name in enumerate(names)}
    row[DAY_KEY_COL] = day_key
    row["missing_count"] = missing_count
    row["data_quality"] = data_quality

    # 固定列顺序。追加模式 header=False 只写值不写列名，必须显式对齐列顺序，
    # 否则 dict 插入顺序（day_key 在末尾）会与既有表头（day_key 在首列）错位，污染 CSV。
    cols = get_feature_columns(track)
    df_row = pd.DataFrame([row])[cols]

    if filepath.exists():
        existing = pd.read_csv(filepath)
        if DAY_KEY_COL in existing.columns and day_key in set(existing[DAY_KEY_COL]):
            # 幂等覆盖：同一天重算时替换旧行，而不是追加出两行同日数据
            existing = existing[existing[DAY_KEY_COL] != day_key]
            combined = pd.concat([existing, df_row], ignore_index=True)
            combined = combined.sort_values(DAY_KEY_COL)[cols]
            combined.to_csv(filepath, index=False)
        else:
            df_row.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_row.to_csv(filepath, index=False)


def load_features_csv(elder_id: str, track: str) -> pd.DataFrame:
    """
    加载某轨的全部特征数据。

    Returns:
        DataFrame，列为 [day_key] + 该轨特征名 + [missing_count, data_quality]
    """
    filepath = get_features_path(elder_id, track)
    if not filepath.exists():
        raise FileNotFoundError(f"Features CSV not found: {filepath}")
    return pd.read_csv(filepath)


def get_feature_vectors(
    elder_id: str,
    start_date: str,
    end_date: str,
    track: str,
) -> np.ndarray:
    """
    获取某轨在指定日期范围内的特征矩阵（按 day_key 升序）。

    Args:
        elder_id: 老人ID
        start_date: 起始 day_key "YYYY-MM-DD"
        end_date: 结束 day_key "YYYY-MM-DD"（含）
        track: "sleep" / "social"

    Returns:
        (n_days, feature_dim) 特征矩阵
    """
    names = get_feature_names(track)
    df = load_features_csv(elder_id, track)
    mask = (df[DAY_KEY_COL] >= start_date) & (df[DAY_KEY_COL] <= end_date)
    df_filtered = df[mask].sort_values(DAY_KEY_COL)

    if len(df_filtered) == 0:
        raise ValueError(
            f"No {track} data found for {elder_id} between {start_date} and {end_date}"
        )

    return df_filtered[names].to_numpy(dtype=np.float64)


def get_daily_vector(elder_id: str, day_key: str, track: str) -> np.ndarray:
    """获取某轨指定 day_key 的单条特征向量 (feature_dim,)"""
    vectors = get_feature_vectors(elder_id, day_key, day_key, track)
    return vectors[0]


def get_recent_vectors(elder_id: str, track: str, days: int = 30) -> np.ndarray:
    """获取某轨最近 N 天的特征矩阵 (n, feature_dim)，按 day_key 升序"""
    names = get_feature_names(track)
    df = load_features_csv(elder_id, track)
    df = df.sort_values(DAY_KEY_COL, ascending=False)
    df_recent = df.head(days).sort_values(DAY_KEY_COL)
    return df_recent[names].to_numpy(dtype=np.float64)


# ========== 基线模型保存/加载 ==========

def get_model_filename(track: str) -> str:
    """某轨的 GRU 权重文件名"""
    return f"gru_{validate_track(track)}.pth"


def get_scaler_path(elder_id: str, track: str) -> Path:
    """某轨的 scaler 路径"""
    return get_baseline_dir(elder_id) / f"scaler_{validate_track(track)}.pkl"


def save_gru_model(model, elder_id: str, filename: str) -> None:
    """保存GRU模型权重"""
    import torch
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dir_path / filename)


def load_gru_model(model_class, elder_id: str, filename: str, **model_kwargs) -> Any:
    """
    加载GRU模型权重。

    model_kwargs 必须传入与训练时一致的 feature_dim / hidden_dim，
    否则 load_state_dict 会因形状不匹配报错——这是有意的，静默加载错误维度的
    模型会让残差完全失去意义。
    """
    import torch
    filepath = get_baseline_dir(elder_id) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model


def save_baseline_meta(elder_id: str, meta: dict) -> None:
    """保存基线元信息到 baseline_meta.json（两轨共用一个文件，按 track 分键）"""
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "baseline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_baseline_meta(elder_id: str) -> dict | None:
    """加载基线元信息；不存在时返回 None"""
    filepath = get_baseline_dir(elder_id) / "baseline_meta.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def update_baseline_meta(elder_id: str, track: str, track_meta: dict) -> None:
    """把某轨的元信息合并进 baseline_meta.json，不影响另一轨已有的键"""
    meta = load_baseline_meta(elder_id) or {}
    meta[validate_track(track)] = track_meta
    save_baseline_meta(elder_id, meta)


def get_track_meta(elder_id: str, track: str) -> dict:
    """取某轨的元信息；缺失时返回空字典"""
    meta = load_baseline_meta(elder_id) or {}
    value = meta.get(validate_track(track))
    return value if isinstance(value, dict) else {}


def save_residual_stats(stats: dict, elder_id: str, track: str) -> None:
    """
    保存某轨的残差统计。

    stats 结构（双残差契约，见 inference.py）：
        {"signed": {"mean": (d,), "std": (d,)},
         "abs":    {"mean": (d,), "std": (d,)}}
    """
    import joblib
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, dir_path / f"residual_stats_{validate_track(track)}.pkl")


def load_residual_stats(elder_id: str, track: str) -> dict:
    """加载某轨的残差统计"""
    import joblib
    filepath = get_baseline_dir(elder_id) / f"residual_stats_{validate_track(track)}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Residual stats not found: {filepath}")
    return joblib.load(filepath)


# ========== 推理日志 ==========

def save_daily_result(elder_id: str, day_key: str, result: dict) -> None:
    """
    保存每日推理结果到JSON文件。

    双轨结果写在同一个文件里：{"sleep": {...}, "social": {...}, ...}
    """
    log_dir = get_log_dir("daily_inference")
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{elder_id}_{day_key}.json"

    # 处理numpy类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(convert(result), f, ensure_ascii=False, indent=2)


def load_daily_results(elder_id: str, n_days: int = 7) -> list[dict]:
    """加载最近N天的推理结果，按 day_key 升序返回"""
    log_dir = get_log_dir("daily_inference")
    if not log_dir.exists():
        return []

    files = sorted(log_dir.glob(f"{elder_id}_*.json"), reverse=True)
    results = []
    for fp in files[:n_days]:
        with open(fp, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return list(reversed(results))  # 按日期升序返回


# ========== 配置加载 ==========

def load_config() -> dict:
    """加载全局配置"""
    import yaml
    config_path = get_project_root() / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_weights() -> dict:
    """加载双轨特征权重配置（原始 JSON 结构）"""
    weights_path = get_project_root() / "config" / "feature_weights.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _track_feature_entries(track: str) -> list[dict]:
    """取某轨在配置文件里的特征条目列表，顺序即配置顺序"""
    cfg = load_feature_weights()
    key = f"{validate_track(track)}_track"
    if key not in cfg:
        raise KeyError(f"feature_weights.json missing section {key!r}")
    entries = cfg[key].get("features")
    if not entries:
        raise KeyError(f"feature_weights.json section {key!r} has no 'features' list")
    return entries


def get_feature_weight_array(track: str) -> np.ndarray:
    """
    获取某轨的权重数组，顺序与该轨 FEATURE_NAMES 严格一致。

    配置缺某个特征时直接报错，不静默填 1.0——权重悄悄变成 1.0 会让加权残差
    偏离设计意图，而且没有任何报错线索。
    """
    names = get_feature_names(track)
    entries = {e["name"]: e for e in _track_feature_entries(track)}

    missing = [n for n in names if n not in entries]
    if missing:
        raise KeyError(
            f"feature_weights.json is missing weights for {track} features: {missing}"
        )

    return np.array([float(entries[n]["weight"]) for n in names], dtype=np.float64)


def get_feature_directions(track: str) -> list[str]:
    """
    获取某轨的异常方向列表（"down" / "up" / "any"），顺序与 FEATURE_NAMES 一致。

    rules.py 用它做方向性判定：down 只认负向超标，up 只认正向超标。
    """
    names = get_feature_names(track)
    entries = {e["name"]: e for e in _track_feature_entries(track)}

    missing = [n for n in names if n not in entries]
    if missing:
        raise KeyError(
            f"feature_weights.json is missing directions for {track} features: {missing}"
        )

    valid = {"down", "up", "any"}
    directions = []
    for n in names:
        d = entries[n].get("direction")
        if d not in valid:
            raise ValueError(
                f"Feature {n!r} has invalid direction {d!r}; expected one of {sorted(valid)}"
            )
        directions.append(d)
    return directions


def get_feature_weight_map(track: str) -> dict[str, float]:
    """获取某轨的 {特征名: 权重} 字典（rules.py 按名字取权重时用）"""
    names = get_feature_names(track)
    arr = get_feature_weight_array(track)
    return {name: float(arr[i]) for i, name in enumerate(names)}

"""
IO工具模块：文件读写、路径管理
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


# ========== 特征数据读写 ==========

def save_daily_features(
    elder_id: str,
    date: str,
    feature_vector: np.ndarray,
    missing_count: int = 0,
    data_quality: str = "valid",
) -> None:
    """
    追加保存每日特征向量到CSV文件。

    Args:
        elder_id: 老人ID
        date: 日期字符串 "YYYY-MM-DD"
        feature_vector: (10,) 特征向量
        missing_count: 缺失特征计数
        data_quality: valid / insufficient / offline
    """
    from src.baseline.scaler_utils import FEATURE_NAMES

    dir_path = get_features_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / "features.csv"

    row = {name: float(feature_vector[i]) for i, name in enumerate(FEATURE_NAMES)}
    row["date"] = date
    row["missing_count"] = missing_count
    row["data_quality"] = data_quality

    # 固定列顺序（与 generate_simulation_data.py 一致）。
    # 追加模式 header=False 只写值不写列名，必须显式对齐列顺序，
    # 否则 dict 插入顺序（date 在末尾）会与既有表头（date 在首列）错位，污染 CSV。
    cols = ["date"] + FEATURE_NAMES + ["missing_count", "data_quality"]
    df_row = pd.DataFrame([row])[cols]

    if filepath.exists():
        df_row.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_row.to_csv(filepath, index=False)


def load_features_csv(elder_id: str) -> pd.DataFrame:
    """
    加载指定老人的全部特征数据。

    Returns:
        DataFrame，列为 FEATURE_NAMES + ['date', 'missing_count', 'data_quality']
    """
    filepath = get_features_dir(elder_id) / "features.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Features CSV not found: {filepath}")
    return pd.read_csv(filepath)


def get_feature_vectors(
    elder_id: str,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    """
    获取指定日期范围内的特征向量。

    Args:
        elder_id: 老人ID
        start_date: 起始日期 "YYYY-MM-DD"
        end_date: 结束日期 "YYYY-MM-DD"（含）

    Returns:
        (n_days, 10) 特征矩阵
    """
    from src.baseline.scaler_utils import FEATURE_NAMES

    df = load_features_csv(elder_id)
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df_filtered = df[mask]

    if len(df_filtered) == 0:
        raise ValueError(f"No data found for {elder_id} between {start_date} and {end_date}")

    return df_filtered[FEATURE_NAMES].to_numpy(dtype=np.float64)


def get_daily_vector(elder_id: str, date: str) -> np.ndarray:
    """
    获取指定日期的单条特征向量。

    Returns:
        (10,) numpy数组
    """
    vectors = get_feature_vectors(elder_id, date, date)
    return vectors[0]


def get_recent_vectors(elder_id: str, days: int = 30) -> np.ndarray:
    """
    获取最近N天的特征向量。

    Args:
        elder_id: 老人ID
        days: 最近天数

    Returns:
        (n, 10) 特征矩阵
    """
    df = load_features_csv(elder_id)
    df = df.sort_values("date", ascending=False)
    df_recent = df.head(days).sort_values("date")

    from src.baseline.scaler_utils import FEATURE_NAMES
    return df_recent[FEATURE_NAMES].to_numpy(dtype=np.float64)


# ========== 基线模型保存/加载 ==========

def save_gru_model(model, elder_id: str, filename: str = "gru.pth") -> None:
    """保存GRU模型权重"""
    import torch
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dir_path / filename)


def load_gru_model(model_class, elder_id: str, filename: str = "gru.pth") -> Any:
    """加载GRU模型权重"""
    import torch
    filepath = get_baseline_dir(elder_id) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = model_class()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model


def save_baseline_meta(elder_id: str, meta: dict) -> None:
    """保存基线元信息（训练日期、训练时 EWMA 样本数等）到 baseline_meta.json"""
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "baseline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_baseline_meta(elder_id: str) -> dict | None:
    """加载基线元信息；不存在时返回 None（兼容旧基线）"""
    filepath = get_baseline_dir(elder_id) / "baseline_meta.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_residual_stats(stats: dict, elder_id: str) -> None:
    """保存残差统计到文件"""
    import joblib
    dir_path = get_baseline_dir(elder_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, dir_path / "residual_stats.pkl")


def load_residual_stats(elder_id: str) -> dict:
    """加载残差统计"""
    import joblib
    filepath = get_baseline_dir(elder_id) / "residual_stats.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Residual stats not found: {filepath}")
    return joblib.load(filepath)


# ========== 推理日志 ==========

def save_daily_result(elder_id: str, date: str, result: dict) -> None:
    """保存每日推理结果到JSON文件"""
    log_dir = get_log_dir("daily_inference")
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{elder_id}_{date}.json"

    # 处理numpy类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
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
    """加载最近N天的推理结果"""
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


def load_feature_weights() -> dict[str, float]:
    """加载特征权重配置"""
    weights_path = get_project_root() / "config" / "feature_weights.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_weight_array() -> np.ndarray:
    """
    获取特征权重数组（与FEATURE_NAMES顺序一致）。

    Returns:
        (10,) 权重数组
    """
    from src.baseline.scaler_utils import FEATURE_NAMES
    weights_dict = load_feature_weights()
    return np.array([weights_dict.get(name, 1.0) for name in FEATURE_NAMES])

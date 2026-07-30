"""
每日推理引擎（双轨）

每日凌晨对每一轨各执行一遍：
    1. 加载该轨的模型、scaler、残差统计、EWMA 池
    2. 获取最近7天 + 今天的该轨特征向量
    3. 归一化 + GRU预测
    4. 计算 signed 与 abs 双残差 → anomaly_score
    5. 动态阈值判断 → is_deviation
    6. 更新对应的 EWMA 池（社交轨按 is_weekend 选池）
    7. 记录日志

★ 双残差契约（本模块最容易写错的地方）

    signed_residual[i] = observed[i] − predicted[i]      ← 判方向
    abs_residual[i]    = |signed_residual[i]|            ← 打幅度分

    anomaly_score = Σ(abs_residual · w) / Σw
    方向判定       = signed_residual 标准化后与 direction 比对

符号约定是 `observed − predicted`（统计学通行定义），好处是名字与符号一致：
实际值**下降** → signed **为负** → direction="down" 判 `signed_z < −threshold`。
反过来的约定（pred − observed）会让"下降"对应正数，是反直觉且极易写反的。

两套统计量都在 calibration/holdout 段估计（见 trainer.py），不用训练集。
"""

from datetime import datetime, timedelta

import numpy as np
import torch

from src.baseline.ewma import TrackEWMAPools
from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import (
    TRACKS,
    get_feature_names,
    transform_data,
    validate_track,
)
from src.utils.io import (
    get_baseline_dir,
    get_daily_vector,
    get_feature_vectors,
    get_feature_weight_array,
    get_model_filename,
    get_scaler_path,
    get_track_meta,
    load_daily_results,
    load_residual_stats,
    save_daily_result,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_weekend(day_key: str) -> bool:
    return datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5


def _normalize_stats(residual_stats: dict, dim: int) -> dict:
    """
    兼容旧格式的残差统计。

    新格式：{"signed": {"mean","std"}, "abs": {"mean","std"}}
    旧格式：{"mean","std"}（abs 残差的统计，无方向信息）

    旧格式下 signed 统计不可得——用 abs 的 std 顶替会造成量纲错配，
    所以这里明确标记 signed 不可用，让上层拒绝做方向判定而不是给出错误结论。
    """
    if "signed" in residual_stats and "abs" in residual_stats:
        return {**residual_stats, "signed_available": True}

    if "mean" in residual_stats and "std" in residual_stats:
        logger.warning(
            "  └─ 检测到旧格式残差统计（仅 abs，无方向信息）。"
            "方向性风险判定将被跳过，请重新建档以获得 signed 统计。"
        )
        abs_stats = {
            "mean": np.asarray(residual_stats["mean"], dtype=np.float64),
            "std": np.asarray(residual_stats["std"], dtype=np.float64),
        }
        return {
            "signed": {"mean": np.zeros(dim), "std": abs_stats["std"]},
            "abs": abs_stats,
            "signed_available": False,
        }

    raise ValueError(f"Unrecognized residual_stats structure: {list(residual_stats)}")


def infer_track(
    elder_id: str,
    day_key: str,
    track: str,
    config: dict | None = None,
) -> dict:
    """
    对某一轨做当日推理。

    Returns:
        {
            "track": str,
            "anomaly_score": float,
            "static_threshold": float,
            "ewma_threshold": float,
            "dynamic_threshold": float,
            "is_deviation": bool,
            "signed_residuals": dict,   # 带符号（判方向）
            "abs_residuals": dict,      # 绝对值（打幅度分）
            "signed_z": dict,           # 标准化后的带符号残差
            "signed_available": bool,
            "ewma_pool": str,
            "status": str,
        }
    """
    track = validate_track(track)

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    risk_cfg = config.get("risk", {})
    ewma_cfg = config.get("ewma", {})
    gru_cfg = config.get("gru", {})
    track_gru = gru_cfg.get(track, {})

    sigma = risk_cfg.get("sigma_multiplier", 2.5)
    cold_start_days = risk_cfg.get("cold_start_observation_days", 0)
    window = gru_cfg.get("window", 7)

    names = get_feature_names(track)
    dim = len(names)
    is_weekend = _is_weekend(day_key)

    base = {
        "track": track,
        "anomaly_score": 0.0,
        "static_threshold": 0.0,
        "ewma_threshold": 0.0,
        "dynamic_threshold": 0.0,
        "is_deviation": False,
        "signed_residuals": {},
        "abs_residuals": {},
        "signed_z": {},
        "signed_available": False,
        "ewma_pool": "weekend" if (track == "social" and is_weekend) else "default",
    }

    # 1. 加载该轨基线文件
    try:
        from src.baseline.scaler_utils import load_scaler
        from src.utils.io import load_gru_model

        scaler = load_scaler(get_scaler_path(elder_id, track))
        model = load_gru_model(
            PersonalBaselineGRU,
            elder_id,
            get_model_filename(track),
            feature_dim=track_gru.get("feature_dim", dim),
            hidden_dim=track_gru.get("hidden_dim", 8),
            num_layers=gru_cfg.get("num_layers", 1),
            dropout=gru_cfg.get("dropout", 0.2),
        )
        residual_stats = _normalize_stats(load_residual_stats(elder_id, track), dim)
    except FileNotFoundError as e:
        logger.info(f"  └─ [{track}] 基线文件缺失，处于冷启动阶段: {e}")
        return {**base, "status": "cold_start"}

    ewma = TrackEWMAPools.load(
        get_baseline_dir(elder_id), track, alpha=ewma_cfg.get("alpha", 0.05)
    )

    # 2. 获取今日特征与过去 window 天特征
    try:
        today_vec = get_daily_vector(elder_id, day_key, track)

        today_dt = datetime.strptime(day_key, "%Y-%m-%d")
        start_dt = today_dt - timedelta(days=window)
        end_dt = today_dt - timedelta(days=1)

        past = get_feature_vectors(
            elder_id,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            track,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  └─ [{track}] 特征数据获取失败: {e}")
        return {**base, "status": "data_insufficient", "error": str(e)}

    if len(past) < window:
        logger.warning(f"  └─ [{track}] 历史数据不足（{len(past)}/{window}天）")
        return {**base, "status": "data_insufficient"}

    past = past[-window:]

    # 3. 归一化
    past_norm = transform_data(scaler, past, track)
    today_norm = transform_data(scaler, today_vec, track)

    # 4. GRU预测
    input_tensor = torch.tensor(
        past_norm.reshape(1, window, dim), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).numpy().flatten()

    # 5. 双残差
    signed_residual = today_norm - pred_norm       # observed − predicted
    abs_residual = np.abs(signed_residual)
    weights = get_feature_weight_array(track)
    anomaly_score = float(np.dot(abs_residual, weights) / np.sum(weights))

    # 6. 阈值：用 abs 统计（与 anomaly_score 同量纲）
    abs_mean = np.asarray(residual_stats["abs"]["mean"], dtype=np.float64)
    abs_std = np.asarray(residual_stats["abs"]["std"], dtype=np.float64)
    base_threshold = float(np.dot(abs_mean, weights) / np.sum(weights))
    std_threshold = float(np.dot(abs_std, weights) / np.sum(weights))
    static_threshold = base_threshold + sigma * std_threshold

    # 动态EWMA阈值：取 min 保持敏感度，防止老人自然衰退后系统变得不敏感。
    # 如果 EWMA 阈值上升（老人状态变差），仍然用较低的 static_threshold 兜底。
    min_samples = ewma.min_samples_required(config, is_weekend=is_weekend)
    pool_n = ewma.n_samples(is_weekend=is_weekend)
    if pool_n >= min_samples:
        ewma_threshold = ewma.get_threshold(sigma, is_weekend=is_weekend)
        dynamic_threshold = min(static_threshold, ewma_threshold)
    else:
        ewma_threshold = static_threshold
        dynamic_threshold = static_threshold

    is_deviation = bool(anomaly_score > dynamic_threshold)

    # 7. 更新对应池并落盘
    ewma.update(anomaly_score, is_weekend=is_weekend)
    ewma.save(get_baseline_dir(elder_id))

    # 8. 标准化 signed 残差：只除以 std（尺度、恒正）以保号，不减均值。
    # signed 残差的均值在留出段上估出来接近 0，减掉意义不大；而若误用 abs 的
    # 正均值去减，会把符号整体拉偏，导致 down 方向永远误触发。
    signed_std = np.asarray(residual_stats["signed"]["std"], dtype=np.float64)
    safe_std = np.where(signed_std < 1e-8, 1e-8, signed_std)
    signed_z = signed_residual / safe_std

    signed_residuals = {n: round(float(signed_residual[i]), 4) for i, n in enumerate(names)}
    abs_residuals = {n: round(float(abs_residual[i]), 4) for i, n in enumerate(names)}
    signed_z_map = {n: round(float(signed_z[i]), 4) for i, n in enumerate(names)}

    # 9. 观察期判定：训练后经过的推理次数（不能直接用样本数，训练已预热 EWMA）
    meta = get_track_meta(elder_id, track)
    n_at_train = meta.get("ewma_n_at_train", 0)
    inferences_since_train = ewma.total_samples() - n_at_train
    in_observation = inferences_since_train <= cold_start_days
    status = "observation" if in_observation else "success"

    return {
        **base,
        "anomaly_score": round(anomaly_score, 4),
        "static_threshold": round(static_threshold, 4),
        "ewma_threshold": round(ewma_threshold, 4),
        "dynamic_threshold": round(dynamic_threshold, 4),
        "is_deviation": is_deviation,
        "signed_residuals": signed_residuals,
        "abs_residuals": abs_residuals,
        "signed_z": signed_z_map,
        "signed_available": residual_stats["signed_available"],
        "ewma_n": pool_n + 1,
        "ewma_min_samples": min_samples,
        "in_observation_period": in_observation,
        "status": status,
    }


def daily_inference(
    elder_id: str,
    day_key: str,
    config: dict | None = None,
    tracks: tuple[str, ...] = TRACKS,
) -> dict:
    """
    双轨每日推理：每轨独立算分与阈值，互不影响。

    Returns:
        {
            "elder_id": str,
            "day_key": str,
            "sleep":  {...},           # infer_track 的返回
            "social": {...},
            "status": str,             # 至少一轨 success 即 success
        }
    """
    logger.info(f"每日推理开始: elder_id={elder_id}, day_key={day_key}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    result: dict = {"elder_id": elder_id, "day_key": day_key}

    for track in tracks:
        result[track] = infer_track(elder_id, day_key, track, config)
        tr = result[track]
        logger.info(
            f"  └─ [{track}] score={tr['anomaly_score']:.4f}, "
            f"threshold={tr['dynamic_threshold']:.4f}, "
            f"deviation={tr['is_deviation']}, status={tr['status']}"
        )

    statuses = {t: result[t]["status"] for t in tracks}
    if any(s in ("success", "observation") for s in statuses.values()):
        overall = "success"
    elif all(s == "cold_start" for s in statuses.values()):
        overall = "cold_start"
    else:
        overall = "data_insufficient"

    result["status"] = overall
    result["track_statuses"] = statuses

    # 连续偏离天数：任一轨偏离即算当日偏离（各轨自己的连续天数由 rules.py 分别统计）
    result["is_deviation"] = any(
        result[t].get("is_deviation", False) for t in tracks
    )

    recent = load_daily_results(elder_id, n_days=7)
    consecutive = 0
    for day_result in reversed(recent):
        if day_result.get("day_key") == day_key:
            continue  # 跳过今天自己的旧记录（重算场景）
        if day_result.get("is_deviation", False):
            consecutive += 1
        else:
            break
    result["consecutive_deviation_days"] = consecutive + (1 if result["is_deviation"] else 0)

    save_daily_result(elder_id, day_key, result)
    return result

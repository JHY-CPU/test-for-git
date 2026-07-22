"""
每日推理引擎

每日凌晨执行：
    1. 加载模型、scaler、残差统计、EWMA
    2. 获取最近7天 + 今天的特征向量
    3. 归一化 + GRU预测
    4. 计算加权残差 → anomaly_score
    5. 动态阈值判断 → is_deviation
    6. 更新EWMA
    7. 记录日志
"""

from datetime import datetime, timedelta

import numpy as np
import torch

from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import FULL_FEATURE_DIM, transform_data
from src.utils.io import (
    get_baseline_dir,
    get_daily_vector,
    get_feature_vectors,
    get_feature_weight_array,
    load_daily_results,
    load_residual_stats,
    save_daily_result,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def daily_inference(
    elder_id: str,
    today_date: str,
    config: dict | None = None,
) -> dict:
    """
    每日推理：检测今日特征是否偏离个人基线。

    Args:
        elder_id: 老人ID
        today_date: 今日日期 "YYYY-MM-DD"
        config: 全局配置字典

    Returns:
        {
            "elder_id": str,
            "date": str,
            "anomaly_score": float,
            "static_threshold": float,
            "ewma_threshold": float,
            "dynamic_threshold": float,
            "is_deviation": bool,
            "feature_residuals": dict,
            "data_quality": str,
            "status": str,  # "success" / "cold_start" / "data_insufficient" / "observation"
            "in_observation_period": bool,  # 是否在冷启动观察期
        }
    """
    logger.info(f"每日推理开始: elder_id={elder_id}, date={today_date}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    risk_cfg = config.get("risk", {})
    ewma_cfg = config.get("ewma", {})
    sigma = risk_cfg.get("sigma_multiplier", 2.5)
    min_dynamic_samples = ewma_cfg.get("min_samples_for_dynamic", 20)
    cold_start_days = risk_cfg.get("cold_start_observation_days", 7)

    # 1. 加载基线文件
    try:
        from src.baseline.scaler_utils import load_scaler
        from src.utils.io import load_gru_model
        scaler = load_scaler(get_baseline_dir(elder_id) / "scaler.pkl")
        model = load_gru_model(PersonalBaselineGRU, elder_id, "gru.pth")
        residual_stats = load_residual_stats(elder_id)
    except FileNotFoundError as e:
        logger.warning(f"  └─ 基线文件缺失，处于冷启动阶段: {e}")
        return {
            "elder_id": elder_id,
            "date": today_date,
            "anomaly_score": 0.0,
            "static_threshold": 0.0,
            "ewma_threshold": 0.0,
            "dynamic_threshold": 0.0,
            "is_deviation": False,
            "feature_residuals": {},
            "data_quality": "cold_start",
            "status": "cold_start",
        }

    # 加载EWMA
    from src.baseline.ewma import CumulativeEWMABaseline
    try:
        ewma = CumulativeEWMABaseline.load(get_baseline_dir(elder_id) / "ewma.pkl")
    except FileNotFoundError:
        ewma = CumulativeEWMABaseline(alpha=ewma_cfg.get("alpha", 0.05))

    # 2. 获取今日特征和过去7天特征
    try:
        today_vec = get_daily_vector(elder_id, today_date)

        # 计算7天前到昨天的日期范围
        today_dt = datetime.strptime(today_date, "%Y-%m-%d")
        start_dt = today_dt - timedelta(days=7)
        end_dt = today_dt - timedelta(days=1)

        past_7 = get_feature_vectors(
            elder_id,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  └─ 特征数据获取失败: {e}")
        return {
            "elder_id": elder_id,
            "date": today_date,
            "anomaly_score": 0.0,
            "status": "data_insufficient",
            "error": str(e),
        }

    # 检查数据量
    if len(past_7) < 7:
        logger.warning(f"  └─ 历史数据不足（{len(past_7)}/7天）")
        return {
            "elder_id": elder_id,
            "date": today_date,
            "anomaly_score": 0.0,
            "status": "data_insufficient",
        }

    if len(past_7) > 7:
        past_7 = past_7[-7:]

    # 3. 归一化
    past_7_norm = transform_data(scaler, past_7)  # (7, 12)
    today_norm = transform_data(scaler, today_vec)  # (12,)

    # 4. GRU预测
    input_tensor = torch.tensor(
        past_7_norm.reshape(1, 7, FULL_FEATURE_DIM), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).numpy().flatten()  # (12,)

    # 5. 计算加权残差
    residual = np.abs(pred_norm - today_norm)
    weights = get_feature_weight_array()
    anomaly_score = float(np.dot(residual, weights) / np.sum(weights))

    # 6. 判断阈值
    base_threshold = float(np.dot(residual_stats["mean"], weights) / np.sum(weights))
    std_threshold = float(np.dot(residual_stats["std"], weights) / np.sum(weights))
    static_threshold = base_threshold + sigma * std_threshold

    # 动态EWMA阈值
    # 逻辑：取 min 保持敏感度，防止老人自然衰退后系统变得不敏感
    # 如果 EWMA 阈值上升（老人状态变差），仍然用较低的 static_threshold 兜底
    if ewma.n >= min_dynamic_samples:
        ewma_threshold = ewma.get_threshold(sigma)
        dynamic_threshold = min(static_threshold, ewma_threshold)
    else:
        ewma_threshold = static_threshold
        dynamic_threshold = static_threshold

    is_deviation = anomaly_score > dynamic_threshold

    # 7. 更新EWMA
    ewma.update(anomaly_score)
    ewma.save(get_baseline_dir(elder_id) / "ewma.pkl")

    # 8. 构建特征残差字典（用于可解释性）
    from src.baseline.scaler_utils import FULL_FEATURE_NAMES
    feature_residuals = {}
    for i, name in enumerate(FULL_FEATURE_NAMES):
        feature_residuals[name] = round(float(residual[i]), 4)

    # 9. 统计连续偏离天数
    recent_results = load_daily_results(elder_id, n_days=7)
    consecutive = 0
    for day_result in reversed(recent_results):
        if day_result.get("is_deviation", False):
            consecutive += 1
        else:
            break

    # 10. 检查是否在冷启动观察期
    # 判断标准：从训练日期（Day 14）开始，后续 cold_start_days 天内
    # 假设训练后的第一天推理即开始计数
    in_observation = ewma.n <= cold_start_days
    final_status = "observation" if in_observation else "success"

    result = {
        "elder_id": elder_id,
        "date": today_date,
        "anomaly_score": round(anomaly_score, 4),
        "static_threshold": round(static_threshold, 4),
        "ewma_threshold": round(ewma_threshold, 4),
        "dynamic_threshold": round(dynamic_threshold, 4),
        "is_deviation": is_deviation,
        "feature_residuals": feature_residuals,
        "consecutive_deviation_days": consecutive + (1 if is_deviation else 0),
        "ewma_n": ewma.n,
        "ewma_mean": round(ewma.mean, 4) if ewma.mean else None,
        "ewma_std": round(ewma.std, 4),
        "data_quality": "valid",
        "status": final_status,
        "in_observation_period": in_observation,
    }

    # 11. 保存结果
    save_daily_result(elder_id, today_date, result)

    logger.info(
        f"  └─ 推理完成: score={anomaly_score:.4f}, "
        f"threshold={dynamic_threshold:.4f}, "
        f"deviation={is_deviation}"
    )

    return result

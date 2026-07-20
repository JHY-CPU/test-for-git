"""
数据完整性校验模块

判断每日数据质量：valid / insufficient / offline
"""

import numpy as np

from src.baseline.scaler_utils import FULL_FEATURE_DIM


def validate_daily_data(
    feature_vector: np.ndarray,
    missing_count: int,
    recent_quality: list[str] | None = None,
) -> str:
    """
    校验每日数据质量。

    Args:
        feature_vector: (12,) 特征向量
        missing_count: 当日无法填充的缺失特征数
        recent_quality: 最近数据质量记录（用于检测离线）

    Returns:
        "valid" - 数据完整，可参与训练和推理
        "insufficient" - ≥3个特征缺失，当日标记为数据不足
        "offline" - 连续≥3天数据不足，触发设备离线
    """
    if feature_vector.shape != (FULL_FEATURE_DIM,):
        raise ValueError(
            f"Expected shape ({FULL_FEATURE_DIM},), got {feature_vector.shape}"
        )

    # 检查数据不足
    if missing_count >= 3:
        # 检查是否连续离线
        if recent_quality is not None and len(recent_quality) >= 3:
            last_3 = recent_quality[-3:]
            if all(q in ("insufficient", "offline") for q in last_3):
                return "offline"
        return "insufficient"

    # 检查极端异常值（传感器故障特征）
    # 负值视为异常
    health_features = feature_vector[:10]
    if np.any(health_features < -100):
        return "insufficient"

    return "valid"


def is_usable_for_training(quality: str) -> bool:
    """检查数据是否可用于模型训练"""
    return quality == "valid"


def is_usable_for_inference(quality: str) -> bool:
    """检查数据是否可用于每日推理"""
    return quality == "valid"  # 仅valid数据参与推理


def get_quality_summary(quality_history: list[str]) -> dict:
    """
    统计数据质量概况。

    Returns:
        {"valid_days": N, "insufficient_days": N, "offline_days": N, "total_days": N}
    """
    return {
        "valid_days": quality_history.count("valid"),
        "insufficient_days": quality_history.count("insufficient"),
        "offline_days": quality_history.count("offline"),
        "total_days": len(quality_history),
        "valid_ratio": quality_history.count("valid") / len(quality_history) if quality_history else 0.0,
    }

"""
Scaler管理工具：为被监测的老人独立维护StandardScaler

确保归一化基准稳定，训练后不重新拟合（防止数据漂移带来的隐藏误差）。

使用 10 维健康特征。
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# 特征名称（10维健康特征）
FEATURE_NAMES = [
    "sad_ratio",            # 悲伤标签占比
    "avg_speed",            # 平均语速
    "pitch_variability",    # 基频变异性（F0标准差，语调单调性↓）
    "distress_events",      # 叹气/哭声频次
    "sleep_efficiency",     # 睡眠效率
    "deep_sleep_ratio",     # 深睡占比
    "sfi",                  # 睡眠碎片化指数
    "hrv_rmssd",            # 心率变异性
    "daily_activity",       # 日间活动量
    "social_turns",         # 社交交互轮次
]

FEATURE_DIM = len(FEATURE_NAMES)  # 10


def create_scaler() -> StandardScaler:
    """创建新的StandardScaler实例"""
    return StandardScaler()


def fit_scaler(scaler: StandardScaler, data: np.ndarray) -> StandardScaler:
    """
    在数据上拟合scaler。

    Args:
        scaler: StandardScaler实例
        data: (n_samples, feature_dim) 特征矩阵

    Returns:
        拟合后的scaler
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features, got {data.shape[1]}"
        )
    return scaler.fit(data)


def transform_data(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """
    使用已有scaler归一化数据（不重新拟合）。

    Args:
        scaler: 已拟合的StandardScaler
        data: (n_samples, feature_dim) 或 (feature_dim,) 特征

    Returns:
        归一化后的数据，保持输入维度
    """
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(1, -1)
    if data.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features, got {data.shape[1]}"
        )
    result = scaler.transform(data)
    return result.flatten() if was_1d else result


def inverse_transform(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """
    反归一化（用于将预测值转回原始尺度）。

    Args:
        scaler: 已拟合的StandardScaler
        data: 归一化后的数据

    Returns:
        原始尺度的数据
    """
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(1, -1)
    return scaler.inverse_transform(data).flatten() if was_1d else scaler.inverse_transform(data)


def save_scaler(scaler: StandardScaler, filepath: str | Path) -> None:
    """保存scaler到文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: str | Path) -> StandardScaler:
    """从文件加载scaler"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")
    return joblib.load(filepath)


def get_scaler_stats(scaler: StandardScaler) -> dict:
    """
    获取scaler的统计信息（用于调试和监控）。

    Returns:
        {"mean": [...], "scale": [...], "var": [...], "n_features": 10}
    """
    return {
        "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "var": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
        "n_features": scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else None,
    }


def check_scaler_fitted(scaler: StandardScaler) -> bool:
    """检查scaler是否已拟合"""
    return hasattr(scaler, "mean_")

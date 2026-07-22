"""
冷启动兜底检测（消除 GRU 就绪前的监测盲区）

GRU 基线需要建档期（默认 14 天）+ 冷启动观察期才能可靠工作。在此之前，
系统若完全不检测，等于头两周处于"盲区"。本模块用简单的"滑动均值 ± Nσ"
统计基线做基础离群检测，作为 GRU 就绪前的兜底。

设计取舍：
    - 只做粗粒度的加权 z-score 判定，不预测、不学习时序节律 —— 这正是它能在
      无模型、无训练的情况下即时可用的原因。
    - sigma 取值比 GRU 轨更保守（默认 3.0），因为滑动基线本身不稳定，宁可漏报
      也不要在建档期就制造误报、动摇用户信任。
    - 与 GRU 轨互斥：一旦 GRU 基线就绪，daily_job 就走正式推理，不再调用本模块。
"""

import numpy as np

from src.baseline.scaler_utils import FEATURE_NAMES


def fallback_deviation_check(
    history: np.ndarray,
    today: np.ndarray,
    weights: np.ndarray,
    sigma: float = 3.0,
) -> dict:
    """
    基于滑动均值/标准差的加权 z-score 离群检测。

    Args:
        history: (n_days, n_features) 今天之前的历史特征（原始量纲，可含 NaN）
        today: (n_features,) 今日特征向量
        weights: (n_features,) 特征权重（与 FEATURE_NAMES 顺序一致）
        sigma: z-score 判偏阈值（越大越保守）

    Returns:
        {
            "anomaly_score": float,   # 加权平均 |z|
            "threshold": float,       # 即 sigma
            "is_deviation": bool,
            "feature_z": dict,        # 各特征的 z-score（可解释性）
            "method": "cold_start_fallback",
        }
    """
    history = np.asarray(history, dtype=np.float64)
    today = np.asarray(today, dtype=np.float64).flatten()

    if history.ndim != 2:
        raise ValueError(f"history must be 2D, got shape {history.shape}")

    n_features = today.shape[0]
    z_scores = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        col = history[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2 or np.isnan(today[j]):
            # 历史样本太少或今日缺失，该特征不参与判定
            continue
        mean = valid.mean()
        std = valid.std()
        if std < 1e-9:
            continue  # 该特征恒定，无法定义偏离
        z_scores[j] = (today[j] - mean) / std

    abs_z = np.abs(z_scores)
    w = np.asarray(weights, dtype=np.float64)
    anomaly_score = float(np.dot(abs_z, w) / np.sum(w)) if np.sum(w) > 0 else 0.0
    is_deviation = anomaly_score > sigma

    feature_z = {
        FEATURE_NAMES[j]: round(float(z_scores[j]), 4)
        for j in range(min(n_features, len(FEATURE_NAMES)))
    }

    return {
        "anomaly_score": round(anomaly_score, 4),
        "threshold": float(sigma),
        "is_deviation": is_deviation,
        "feature_z": feature_z,
        "method": "cold_start_fallback",
    }

"""
训练数据健康门禁（防止 GRU "学坏"）

冷启动 / 微调用的训练数据若本身混入异常态（老人建档期恰好状态不好），
GRU 会把异常学成"正常基线"，之后再也报不出来。本模块在训练前对数据做
逐特征离群筛查，识别出可疑的异常天。

方法：MAD（中位数绝对偏差）。相比均值 ± Nσ，中位数与 MAD 对离群点本身
不敏感，适合"样本很少、又要判断哪几天离群"的冷启动场景。

    modified_z = 0.6745 * (x - median) / MAD

|modified_z| > 阈值（默认 3.5）视为该特征离群。一天中有 ≥ min_bad_features
个特征离群，则整天判为离群天。
"""

import numpy as np

from src.baseline.scaler_utils import FEATURE_NAMES

# Consistency constant: 0.6745 = Φ⁻¹(0.75)，使 MAD 在正态分布下等价于标准差
_MAD_SCALE = 0.6745


def detect_outlier_days(
    data: np.ndarray,
    z_threshold: float = 3.5,
    min_bad_features: int = 2,
) -> dict:
    """
    逐特征 MAD 离群检测，识别可疑异常天。

    Args:
        data: (n_days, n_features) 特征矩阵（原始量纲，允许含 NaN）
        z_threshold: modified z-score 绝对值阈值，超过视为该特征离群
        min_bad_features: 一天中离群特征数达到此值则整天判为离群

    Returns:
        {
            "outlier_day_indices": list[int],   # 离群天在 data 中的行索引
            "outlier_ratio": float,             # 离群天占比
            "per_day_bad_counts": list[int],    # 每天的离群特征数
            "feature_flags": np.ndarray(bool),  # (n_days, n_features) 每格是否离群
            "n_days": int,
        }
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_days, n_features), got shape {data.shape}")

    n_days, n_features = data.shape
    feature_flags = np.zeros((n_days, n_features), dtype=bool)

    for j in range(n_features):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            # 有效样本太少，无法可靠估计中位数/MAD，跳过该特征
            continue

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))

        if mad < 1e-9:
            # 该特征几乎恒定：用均值绝对偏差兜底，避免除零
            mean_abs_dev = np.mean(np.abs(valid - median))
            if mean_abs_dev < 1e-9:
                continue  # 完全恒定，不可能有离群
            modified_z = 0.7979 * (col - median) / mean_abs_dev
        else:
            modified_z = _MAD_SCALE * (col - median) / mad

        # NaN 不参与离群判定（缺失由 imputer/validator 另行处理）
        flags = np.abs(modified_z) > z_threshold
        flags[np.isnan(col)] = False
        feature_flags[:, j] = flags

    per_day_bad_counts = feature_flags.sum(axis=1).tolist()
    outlier_day_indices = [
        i for i, c in enumerate(per_day_bad_counts) if c >= min_bad_features
    ]
    outlier_ratio = len(outlier_day_indices) / n_days if n_days > 0 else 0.0

    return {
        "outlier_day_indices": outlier_day_indices,
        "outlier_ratio": outlier_ratio,
        "per_day_bad_counts": per_day_bad_counts,
        "feature_flags": feature_flags,
        "n_days": n_days,
    }


def describe_outlier_days(data: np.ndarray, report: dict) -> list[str]:
    """
    把离群检测结果转成人类可读的说明（用于日志）。

    Args:
        data: 与 detect_outlier_days 相同的 (n_days, n_features) 矩阵
        report: detect_outlier_days 的返回值

    Returns:
        每个离群天一行的描述字符串列表
    """
    lines = []
    flags = report["feature_flags"]
    for i in report["outlier_day_indices"]:
        bad_feats = [
            FEATURE_NAMES[j] for j in range(flags.shape[1])
            if j < len(FEATURE_NAMES) and flags[i, j]
        ]
        lines.append(f"Day#{i}: 离群特征={bad_feats}")
    return lines

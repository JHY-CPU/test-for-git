"""
冷启动兜底检测（消除 GRU 就绪前的监测盲区）

GRU 基线需要建档期（默认 35 天）才能可靠工作。在此之前，
系统若完全不检测，等于头五周处于"盲区"。本模块用稳健统计基线
（中位数 / MAD）做基础离群检测，作为 GRU 就绪前的兜底。按轨独立运行。

设计取舍：
    - 只做粗粒度的加权 z-score 判定，不预测、不学习时序节律 —— 这正是它能在
      无模型、无训练的情况下即时可用的原因。
    - sigma 取值比 GRU 轨更保守（默认 3.0），因为滑动基线本身不稳定，宁可漏报
      也不要在建档期就制造误报、动摇用户信任。
    - 与 GRU 轨互斥：某轨的 GRU 基线就绪后，daily_job 就走该轨的正式推理。

★ 相比旧版修掉的三个缺陷（旧版是"滑动均值 ± Nσ"）：

  1. 缺失维稀释异常分
     旧：`anomaly = Σ(|z|·w) / Σ(全部w)`，跳过的维 z 留 0 但权重仍计入分母
         → 缺的维越多，分越低，越不容易报警。传感器坏了反而更"正常"。
     新：分母只累加有效维的权重。

  2. std ≈ 0 时直接 continue
     旧：`if std < 1e-9: continue` → 漏掉"过去恒定、今天突变"，
         而这恰恰是最该报的情形。
     新：改用 MAD 下限判定——历史恒定时，只要今天偏离中位数超过下限的 k 倍就算偏离。

  3. 用 np.abs 丢方向
     旧：只返回 |z|，rules.py 无法判 up/down 方向性。
     新：同时返回 signed 与 abs 两套 z，与 GRU 轨的双残差契约同接口。

改用中位数/MAD 而非均值/标准差：建档期只有十几个样本，若其中混入 1-2 个异常天，
均值和标准差会被拽走（异常天把 std 撑大 → 后续真异常反而检不出）。
中位数与 MAD 对离群点本身不敏感，更适合"样本少且可能不干净"的冷启动场景。
"""

import numpy as np

from src.baseline.scaler_utils import get_feature_names

# 0.6745 = Φ⁻¹(0.75)，使 MAD 在正态分布下等价于标准差
_MAD_SCALE = 0.6745

# MAD 的绝对下限，按特征各自的量纲比例给。历史完全恒定时用它兜底，
# 避免分母为 0；同时也防止极小的 MAD 把正常波动放大成巨大 z 分。
_MAD_FLOOR_RATIO = 0.02   # 取 |median| 的 2%
_MAD_FLOOR_ABS = 1e-6     # median 也为 0 时的绝对地板


def _robust_scale(valid: np.ndarray) -> float:
    """
    该特征的稳健尺度（相当于标准差的角色）。

    MAD 为 0（历史恒定或半数以上取同值）时退回平均绝对偏差，
    仍为 0 则用 |median| 的比例地板 —— 不再像旧版那样直接跳过该维。
    """
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    if mad > 1e-12:
        return mad / _MAD_SCALE

    mean_abs_dev = float(np.mean(np.abs(valid - median)))
    if mean_abs_dev > 1e-12:
        # 0.7979 = sqrt(2/π)，使平均绝对偏差在正态下等价于标准差
        return mean_abs_dev / 0.7979

    # 历史完全恒定：用量纲比例地板，让"今天突变"仍能被检出
    return max(abs(median) * _MAD_FLOOR_RATIO, _MAD_FLOOR_ABS)


def fallback_deviation_check(
    history: np.ndarray,
    today: np.ndarray,
    weights: np.ndarray,
    track: str,
    sigma: float = 3.0,
) -> dict:
    """
    基于中位数/MAD 稳健基线的加权 z-score 离群检测。

    Args:
        history: (n_days, n_features) 今天之前的历史特征（原始量纲，可含 NaN）
        today: (n_features,) 今日特征向量
        weights: (n_features,) 特征权重（与该轨 FEATURE_NAMES 顺序一致）
        track: "sleep" / "social"
        sigma: z-score 判偏阈值（越大越保守）

    Returns:
        {
            "anomaly_score": float,       # 加权平均 |z|，分母只算有效维
            "threshold": float,           # 即 sigma
            "is_deviation": bool,
            "feature_z": dict,            # 带符号 z（供方向性判定）
            "feature_z_abs": dict,        # 绝对值 z（供幅度打分）
            "valid_features": list[str],  # 参与判定的特征
            "skipped_features": list[str],# 未参与判定的特征（历史或今日缺失）
            "method": "cold_start_fallback",
        }
    """
    names = get_feature_names(track)

    history = np.asarray(history, dtype=np.float64)
    today = np.asarray(today, dtype=np.float64).flatten()

    if history.ndim != 2:
        raise ValueError(f"history must be 2D, got shape {history.shape}")
    if today.shape[0] != len(names):
        raise ValueError(
            f"Track {track!r} expects {len(names)} features, got {today.shape[0]}"
        )
    if history.shape[1] != len(names):
        raise ValueError(
            f"history has {history.shape[1]} columns, expected {len(names)}"
        )

    n_features = today.shape[0]
    signed_z = np.full(n_features, np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_features, dtype=bool)

    for j in range(n_features):
        col = history[:, j]
        valid = col[~np.isnan(col)]
        # 至少 3 个历史点才能谈"中位数与 MAD"；今日缺失也无从比较
        if len(valid) < 3 or np.isnan(today[j]):
            continue

        median = float(np.median(valid))
        scale = _robust_scale(valid)
        signed_z[j] = (today[j] - median) / scale
        valid_mask[j] = True

    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != n_features:
        raise ValueError(f"weights has {w.shape[0]} entries, expected {n_features}")

    # ★ 缺陷 1 的修法：分母只累加有效维的权重。
    # 旧版用 Σ(全部w) 做分母，跳过的维以 z=0 计入 → 缺的维越多分越低，
    # 传感器故障会让系统显得更"正常"，这与告警的目的正好相反。
    valid_w_sum = float(w[valid_mask].sum())
    if valid_w_sum > 0:
        abs_z = np.abs(signed_z[valid_mask])
        anomaly_score = float(np.dot(abs_z, w[valid_mask]) / valid_w_sum)
    else:
        anomaly_score = 0.0

    is_deviation = bool(valid_w_sum > 0 and anomaly_score > sigma)

    feature_z = {
        names[j]: round(float(signed_z[j]), 4)
        for j in range(n_features) if valid_mask[j]
    }
    feature_z_abs = {name: abs(value) for name, value in feature_z.items()}

    return {
        "anomaly_score": round(anomaly_score, 4),
        "threshold": float(sigma),
        "is_deviation": is_deviation,
        "feature_z": feature_z,
        "feature_z_abs": feature_z_abs,
        "valid_features": [names[j] for j in range(n_features) if valid_mask[j]],
        "skipped_features": [names[j] for j in range(n_features) if not valid_mask[j]],
        "method": "cold_start_fallback",
    }

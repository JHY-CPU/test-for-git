"""
缺失值处理模块（按轨独立）

策略：
    - 单日单个特征缺失 → 前向填充（取昨日值）
    - 连续缺失≤3天 → 前向填充
    - 连续缺失>3天 → 线性插值（如有前后数据）或标记为质量降级
    - 某轨单日≥3个特征缺失 → 该轨标记"数据不足"
    - 连续≥3天数据不足 → 触发"设备离线"告警

例外：copresence_min 禁止前向填充。其它特征反映老人自身的行为习惯，
昨天的值对今天有预测力；而"今天有没有人来"取决于子女的安排，
用昨天填今天等于凭空伪造社会接触。
"""

import numpy as np

from src.baseline.scaler_utils import get_feature_dim, get_feature_names
from src.data_pipeline.aggregator import NO_FORWARD_FILL_FEATURES


def impute_missing(
    current_vec: np.ndarray,
    track: str,
    prev_day_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, int, list[str]]:
    """
    对某轨当日特征向量执行前向填充。

    Args:
        current_vec: (feature_dim,) 当日特征向量（含NaN）
        track: "sleep" / "social"
        prev_day_vec: (feature_dim,) 昨日特征向量（用于填充），None时无法填充

    Returns:
        (filled_vector, missing_count, missing_features)
        - filled_vector: 填充后的向量
        - missing_count: 无法填充的特征数（含禁止填充却缺失的特征）
        - missing_features: 无法填充的特征名列表
    """
    names = get_feature_names(track)
    dim = get_feature_dim(track)

    if current_vec.shape != (dim,):
        raise ValueError(
            f"Track {track!r} expects shape ({dim},), got {current_vec.shape}"
        )

    missing_mask = np.isnan(current_vec)
    filled = current_vec.copy()

    # 禁止前向填充的特征：缺了就是缺了，不借昨天的值
    no_fill_idx = {i for i, n in enumerate(names) if n in NO_FORWARD_FILL_FEATURES}

    if missing_mask.any() and prev_day_vec is not None:
        # 前向填充：用昨日值替换 NaN。对老年人的日尺度生理/行为特征，"今天大概率接近昨天"
        # 是比填均值更稳妥的假设——填均值会把偏离往群体中心拉、削弱个人基线的偏离信号。
        for i in range(dim):
            if i in no_fill_idx:
                continue
            if missing_mask[i] and not np.isnan(prev_day_vec[i]):
                filled[i] = prev_day_vec[i]
                missing_mask[i] = False

    # 仍填不上的填 0。归一化空间里 0 ≈ 训练均值，是"信息中性"的占位，
    # 不会伪造出一个偏离；同时这些特征已计入 final_missing_count，由上层据此降级数据质量。
    filled[missing_mask] = 0.0

    # 统计最终无法填充的特征
    final_missing_list = []
    for i in range(dim):
        if not np.isnan(current_vec[i]):
            continue
        if i in no_fill_idx:
            # 禁止填充的特征只要原始缺失就永远计入缺失
            final_missing_list.append(names[i])
        elif prev_day_vec is None or np.isnan(prev_day_vec[i]):
            final_missing_list.append(names[i])

    return filled, len(final_missing_list), final_missing_list


def impute_sequence(
    feature_sequence: np.ndarray,
    track: str,
    max_forward_days: int = 3,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    对某轨特征序列进行智能填充（前向填充 + 线性插值）。

    对于连续缺失超过max_forward_days的特征，如果前后都有数据，使用线性插值。

    Args:
        feature_sequence: (n_days, feature_dim) 特征序列
        track: "sleep" / "social"
        max_forward_days: 前向填充的最大天数，超过则尝试插值

    Returns:
        (filled_sequence, degraded_features)
        - filled_sequence: 填充后的序列
        - degraded_features: 各特征的质量降级天数统计
    """
    names = get_feature_names(track)
    dim = get_feature_dim(track)

    if feature_sequence.ndim != 2 or feature_sequence.shape[1] != dim:
        raise ValueError(
            f"Track {track!r} expects (n_days, {dim}), got {feature_sequence.shape}"
        )

    n_days = feature_sequence.shape[0]
    filled = feature_sequence.copy()
    degraded_features = {name: 0 for name in names}

    for feat_idx in range(dim):
        feature_col = filled[:, feat_idx]
        missing_mask = np.isnan(feature_col)

        if not missing_mask.any():
            continue

        # 找出连续缺失段
        i = 0
        while i < n_days:
            if missing_mask[i]:
                start = i
                while i < n_days and missing_mask[i]:
                    i += 1
                end = i - 1
                gap_length = end - start + 1

                if start > 0 and not np.isnan(feature_col[start - 1]):
                    fill_value = feature_col[start - 1]
                    if gap_length <= max_forward_days:
                        # 短缺失：前向填充
                        feature_col[start:end + 1] = fill_value
                    else:
                        # 长缺失：尝试线性插值
                        if end < n_days - 1 and not np.isnan(feature_col[end + 1]):
                            next_value = feature_col[end + 1]
                            interp_values = np.linspace(
                                fill_value, next_value, gap_length + 2
                            )[1:-1]
                            feature_col[start:end + 1] = interp_values
                            degraded_features[names[feat_idx]] += gap_length
                        else:
                            feature_col[start:end + 1] = fill_value
                            degraded_features[names[feat_idx]] += gap_length
                else:
                    # 无前置数据，填充0
                    feature_col[start:end + 1] = 0.0
                    degraded_features[names[feat_idx]] += gap_length
            else:
                i += 1

        filled[:, feat_idx] = feature_col

    return filled, degraded_features


def check_offline_status(
    recent_quality: list[str],
    threshold: int = 3,
) -> bool:
    """
    检查是否应触发"设备离线"告警。

    Args:
        recent_quality: 最近N天的数据质量标记列表
        threshold: 连续数据不足天数阈值，默认3天

    Returns:
        True表示应触发离线告警
    """
    if len(recent_quality) < threshold:
        return False

    for quality in recent_quality[-threshold:]:
        if quality == "valid":
            return False

    return True

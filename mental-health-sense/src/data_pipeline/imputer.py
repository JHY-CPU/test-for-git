"""
缺失值处理模块

策略：
    - 单日单个特征缺失 → 前向填充（取昨日值）
    - 连续缺失≤3天 → 前向填充
    - 连续缺失>3天 → 线性插值（如有前后数据）或标记为质量降级
    - 单日≥3个特征缺失 → 标记"数据不足"
    - 连续≥3天数据不足 → 触发"设备离线"告警
"""

import numpy as np

from src.baseline.scaler_utils import FULL_FEATURE_DIM, FULL_FEATURE_NAMES


def impute_missing(
    current_vec: np.ndarray,
    prev_day_vec: np.ndarray | None = None,
    forward_fill_only: bool = True,
) -> tuple[np.ndarray, int, list[str]]:
    """
    对当前日特征向量执行前向填充。

    Args:
        current_vec: (10,) 当日特征向量（含NaN）
        prev_day_vec: (10,) 昨日特征向量（用于填充），None时无法填充
        forward_fill_only: 是否仅使用前向填充（默认True，保持向后兼容）

    Returns:
        (filled_vector, missing_count, missing_features)
        - filled_vector: 填充后的(10,)向量
        - missing_count: 无法填充的特征数
        - missing_features: 无法填充的特征名称列表
    """
    if current_vec.shape != (FULL_FEATURE_DIM,):
        raise ValueError(
            f"Expected shape ({FULL_FEATURE_DIM},), got {current_vec.shape}"
        )

    # 全部10维都是健康特征
    health_features = current_vec

    # 找出缺失位置
    missing_mask = np.isnan(health_features)
    missing_count = int(missing_mask.sum())

    filled = health_features.copy()

    if missing_count > 0 and prev_day_vec is not None:
        # 前向填充：用昨日值替换NaN
        prev_health = prev_day_vec
        for i in range(FULL_FEATURE_DIM):
            if missing_mask[i] and not np.isnan(prev_health[i]):
                filled[i] = prev_health[i]
                missing_mask[i] = False

    # 对仍未填充的，用0填充（标记为数据不足）
    filled[missing_mask] = 0.0

    # 统计最终无法填充的特征
    final_missing_count = 0
    final_missing_list = []
    for i in range(FULL_FEATURE_DIM):
        if np.isnan(health_features[i]):
            if prev_day_vec is None or np.isnan(prev_day_vec[i]):
                final_missing_count += 1
                final_missing_list.append(FULL_FEATURE_NAMES[i])

    return filled, final_missing_count, final_missing_list


def impute_sequence(
    feature_sequence: np.ndarray,
    max_forward_days: int = 3,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    对特征序列进行智能填充（前向填充 + 线性插值）。

    对于连续缺失超过max_forward_days的特征，如果前后都有数据，使用线性插值。

    Args:
        feature_sequence: (n_days, 10) 特征序列
        max_forward_days: 前向填充的最大天数，超过则尝试插值

    Returns:
        (filled_sequence, degraded_features)
        - filled_sequence: 填充后的序列
        - degraded_features: 各特征的质量降级天数统计
    """
    n_days = feature_sequence.shape[0]
    filled = feature_sequence.copy()
    degraded_features = {name: 0 for name in FULL_FEATURE_NAMES}

    for feat_idx in range(FULL_FEATURE_DIM):  # 处理所有10维健康特征
        feature_col = filled[:, feat_idx]
        missing_mask = np.isnan(feature_col)

        if not missing_mask.any():
            continue

        # 找出连续缺失段
        i = 0
        while i < n_days:
            if missing_mask[i]:
                # 找出连续缺失段的起止
                start = i
                while i < n_days and missing_mask[i]:
                    i += 1
                end = i - 1
                gap_length = end - start + 1

                # 前向填充
                if start > 0 and not np.isnan(feature_col[start - 1]):
                    fill_value = feature_col[start - 1]
                    if gap_length <= max_forward_days:
                        # 短缺失：前向填充
                        feature_col[start:end+1] = fill_value
                    else:
                        # 长缺失：尝试线性插值
                        if end < n_days - 1 and not np.isnan(feature_col[end + 1]):
                            # 有后续数据，线性插值
                            next_value = feature_col[end + 1]
                            interp_values = np.linspace(fill_value, next_value, gap_length + 2)[1:-1]
                            feature_col[start:end+1] = interp_values
                            degraded_features[FULL_FEATURE_NAMES[feat_idx]] += gap_length
                        else:
                            # 无后续数据，前向填充
                            feature_col[start:end+1] = fill_value
                            degraded_features[FULL_FEATURE_NAMES[feat_idx]] += gap_length
                else:
                    # 无前置数据，填充0
                    feature_col[start:end+1] = 0.0
                    degraded_features[FULL_FEATURE_NAMES[feat_idx]] += gap_length
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
                        (["valid", "insufficient", "insufficient", ...])
        threshold: 连续数据不足天数阈值，默认3天

    Returns:
        True表示应触发离线告警
    """
    if len(recent_quality) < threshold:
        return False

    # 检查最近threshold天是否全为 "insufficient" 或 "offline"
    for quality in recent_quality[-threshold:]:
        if quality == "valid":
            return False

    return True

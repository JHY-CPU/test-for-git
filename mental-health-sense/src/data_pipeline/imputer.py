"""
缺失值处理模块（按轨独立）

策略：
    - 单日单个特征缺失 → 前向填充（取昨日值）
    - 某轨单日≥3个特征缺失 → 该轨标记"数据不足"

数据质量四态与"设备离线"判据归 validator 管（validate_daily_data /
check_prolonged_degradation），本模块只负责填充。

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


# 已删除（2026-07-31）：
#
#   impute_sequence()      —— 序列级前向填充 + 线性插值。零调用方：每日链路走的是
#       单日的 impute_missing，训练/推理都只读 data_quality=="valid" 的行，从来
#       不需要对整段序列做填充。属推测性通用化，需要时从 git 历史取回。
#
#   check_offline_status() —— 与 validator.check_prolonged_degradation 是逐行
#       等价的重复实现（同样是"末尾 threshold 条全部 != valid"），只有默认值
#       不同（3 vs 5）。两份会漂的"离线判据"比没有更危险：改了一处忘另一处，
#       界面与告警就会给出互相矛盾的结论。统一保留 validator 那份——
#       四态数据质量的定义本来就归 validator 管。

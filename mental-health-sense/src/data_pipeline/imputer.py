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

    # ★ 仍填不上的**保持 NaN**，绝不填 0。
    #
    # 旧实现这里是 `filled[missing_mask] = 0.0`，注释写的是"归一化空间里 0 ≈ 训练
    # 均值，是信息中性的占位"。但 filled 是**原始量纲**向量，会被原样写进
    # features_{track}.csv，归一化要等到推理层用冻结的 scaler 才发生。
    # 于是"信息中性"的论证在错误的空间里成立，实际效果是伪造出极端偏离。
    # 用本仓 E001 自己的数据实测：
    #
    #     night_hr_mean    原始 0 → z = −15.2
    #     sleep_efficiency 原始 0 → z = −11.8
    #     rar_amplitude    原始 0 → z = −5.6
    #     copresence_min   原始 0 → z = −1.44   （足以让 _exceeds(z,"down",1.0) 为真）
    #
    # 两个具体后果：
    #   1. copresence_min 是 social_decline 权重最高的**必选维**，设计上"缺了它
    #      规则就永远不触发"（validator 的 CRITICAL_FEATURES 也是这个意思），
    #      现在反而被免费送上一个满足方向的 z。"测不到"变成了"确实没人来"。
    #   2. 缺 1 维时数据质量仍判 valid（DEGRADED_THRESHOLD=2），于是这一行
    #      −15σ 的点会进 StandardScaler.fit 与 GRU 训练集，把归一化基准带偏，
    #      之后所有 signed_z 的分母都是错的。
    #
    # NaN 才是诚实的表示，而且是本仓已有的词汇：cold_start_fallback 用
    # np.isnan 判"这维今天测不到"，产出 valid_features / skipped_features。
    # 推理层据此把该维排除出打分（见 inference.infer_track）。
    filled[missing_mask] = np.nan

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

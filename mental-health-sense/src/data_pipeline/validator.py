"""
数据完整性校验模块（按轨独立）

每日数据质量四态：
    valid        数据完整，可参与训练和推理
    degraded     数据可疑（缺 2 维 / 设备电量低），仍进模型但不计入连续偏离天数
    insufficient 缺 ≥3 维，该轨当日作废
    offline      连续 ≥3 天数据不足，触发设备离线告警

为什么要有 degraded 这一档：缺 1 维前向填充后基本可信，缺 2 维就够可疑了——
但直接作废又太浪费。折中是"照常进模型算分，但不让它攒进连续偏离天数"，
这样传感器抖动不会凭空凑出一个连续 3 天的预警。
"""

import numpy as np

from src.baseline.scaler_utils import get_feature_dim, get_feature_names
from src.data_pipeline.aggregator import MISSING_THRESHOLD

# 数据质量枚举（顺序即严重程度递增）
QUALITY_VALID = "valid"
QUALITY_DEGRADED = "degraded"
QUALITY_INSUFFICIENT = "insufficient"
QUALITY_OFFLINE = "offline"

# 已删除（2026-07-31）：QUALITY_ENUM 元组，零引用（只在某个 docstring 的文本里
# 被提到过）。四个档位常量本身是活的，各处按名引用。

# 缺 2 维即降级（缺 1 维前向填充后仍算 valid）
DEGRADED_THRESHOLD = 2

# ★ 关键特征：缺失时整轨至少降级，不看缺失个数。
#
# copresence_min 是社会连接轨**唯一**的社会接触指标（其余 4 维测的是活动量与节律）。
# 它缺失时若仍按"只缺 1 维"算 valid，系统会拿 4 个活动/节律维继续输出
# "社会连接正常"——而这 4 维根本测不到"跟人在一起"这件事。
# 同时「社会连接减弱」规则要求 copresence_min 必须方向性超标，缺了它规则永远不触发，
# 于是"测不到"被静默呈现为"没问题"。降级是为了让这个区别在数据里留下痕迹。
#
# 不直接判 insufficient 的理由：剩下 4 维对活动/节律仍有效，作废太浪费。
# degraded 的语义（进模型算分、但不计入连续偏离天数）正好合适。
CRITICAL_FEATURES: dict[str, frozenset[str]] = {
    "social": frozenset({"copresence_min"}),
    "sleep": frozenset(),   # 睡眠轨没有单一不可替代维，按缺失个数判即可
}

# 进入模型的质量档
_USABLE_FOR_INFERENCE = frozenset({QUALITY_VALID, QUALITY_DEGRADED})
# 计入连续偏离天数的质量档：degraded 不计入
_COUNTS_TOWARD_CONSECUTIVE = frozenset({QUALITY_VALID})


def validate_daily_data(
    feature_vector: np.ndarray,
    missing_count: int,
    track: str,
    recent_quality: list[str] | None = None,
    device_degraded: bool = False,
    missing_features: list[str] | None = None,
) -> str:
    """
    校验某轨的每日数据质量。

    Args:
        feature_vector: (feature_dim,) 特征向量
        missing_count: 当日无法填充的缺失特征数
        track: "sleep" / "social"
        recent_quality: 最近数据质量记录（用于检测离线）
        device_degraded: 外部传入的设备降级信号（如 T1C 电量 <20%）
        missing_features: 缺失特征名列表；提供时会检查 CRITICAL_FEATURES

    Returns:
        QUALITY_ENUM 之一
    """
    dim = get_feature_dim(track)
    if feature_vector.shape != (dim,):
        raise ValueError(
            f"Track {track!r} expects shape ({dim},), got {feature_vector.shape}"
        )

    # 检查数据不足
    if missing_count >= MISSING_THRESHOLD:
        # 检查是否连续离线
        if recent_quality is not None and len(recent_quality) >= 3:
            last_3 = recent_quality[-3:]
            if all(q in (QUALITY_INSUFFICIENT, QUALITY_OFFLINE) for q in last_3):
                return QUALITY_OFFLINE
        return QUALITY_INSUFFICIENT

    # 检查极端异常值（传感器故障特征）
    if np.any(feature_vector < -100):
        return QUALITY_INSUFFICIENT

    # ★ 关键特征缺失 → 至少降级（不看缺失个数，见 CRITICAL_FEATURES 的说明）
    if missing_features and missing_critical_features(track, missing_features):
        return QUALITY_DEGRADED

    # 缺 2 维或设备自报降级 → degraded
    if missing_count >= DEGRADED_THRESHOLD or device_degraded:
        return QUALITY_DEGRADED

    return QUALITY_VALID


def missing_critical_features(track: str, missing_features: list[str]) -> list[str]:
    """该轨缺失的关键特征列表（空列表表示关键特征齐全）"""
    critical = CRITICAL_FEATURES.get(track, frozenset())
    return [f for f in missing_features if f in critical]


def describe_track_capability(track: str, missing_features: list[str]) -> str | None:
    """
    关键特征缺失时，返回一句能力受限说明，供日志与对外文案使用。

    对外必须明示"测不到"而非沉默——把"设备没数据"呈现为"一切正常"是最危险的失效模式。
    """
    lost = missing_critical_features(track, missing_features)
    if not lost:
        return None
    if track == "social" and "copresence_min" in lost:
        return (
            "社会接触指标不可用（copresence_min 缺失），本日仅能评估活动量与作息节律，"
            "不对『社会接触』作结论"
        )
    return f"关键特征缺失 {lost}，该轨能力受限"


def is_usable_for_training(quality: str) -> bool:
    """检查数据是否可用于模型训练（只用最干净的数据建基线）"""
    return quality == QUALITY_VALID


def is_usable_for_inference(quality: str) -> bool:
    """检查数据是否可用于每日推理（degraded 仍出结果，只是不攒连续天数）"""
    return quality in _USABLE_FOR_INFERENCE


def counts_toward_consecutive(quality: str) -> bool:
    """该质量档是否计入"连续偏离天数"。

    degraded / insufficient / offline 都不计入——否则传感器抖动会被攒成预警。
    """
    return quality in _COUNTS_TOWARD_CONSECUTIVE


def get_quality_summary(quality_history: list[str]) -> dict:
    """
    统计数据质量概况。

    Returns:
        {"valid_days": N, "degraded_days": N, "insufficient_days": N,
         "offline_days": N, "total_days": N, "valid_ratio": float}
    """
    total = len(quality_history)
    return {
        "valid_days": quality_history.count(QUALITY_VALID),
        "degraded_days": quality_history.count(QUALITY_DEGRADED),
        "insufficient_days": quality_history.count(QUALITY_INSUFFICIENT),
        "offline_days": quality_history.count(QUALITY_OFFLINE),
        "total_days": total,
        "valid_ratio": quality_history.count(QUALITY_VALID) / total if total else 0.0,
    }


def check_prolonged_degradation(
    quality_history: list[str],
    threshold: int = 5,
) -> bool:
    """
    连续 ≥threshold 天非 valid → 应升级为运维告警。

    这条是为了让"长期降级"和"长期正常"在界面上可区分：传感器坏了 5 天，
    界面不该显示"一切正常"，而要明示"当前处于数据不足状态"。
    """
    if len(quality_history) < threshold:
        return False
    return all(q != QUALITY_VALID for q in quality_history[-threshold:])

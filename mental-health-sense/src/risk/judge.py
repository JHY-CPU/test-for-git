"""
风险等级判定

基于连续超标天数 + 风险聚合分数，判定四级风险：
    0 = 正常
    1 = 关注
    2 = 提醒
    3 = 严重

防误报机制：要求连续超标而非单点触发。
"""

import numpy as np

from src.risk.rules import classify_risk_type, RISK_RULES
from src.utils.io import load_daily_results, load_residual_stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


def judge_risk_level(
    elder_id: str,
    daily_results: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """
    判定当前风险等级。

    输入：近7天的每日推理结果
    输出：当前风险等级及详细报告

    Args:
        elder_id: 老人ID
        daily_results: 近7天推理结果（可选，不提供则自动加载）
        config: 全局配置

    Returns:
        {
            "elder_id": str,
            "risk_level": int,       # 0/1/2/3
            "risk_label": str,       # "正常"/"关注"/"提醒"/"严重"
            "risk_types": list[dict], # 活跃的风险类型
            "consecutive_deviation": int,
            "avg_anomaly_7d": float,
            "max_anomaly_7d": float,
            "recommendation": str,
        }
    """
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    risk_cfg = config.get("risk", {})
    consecutive_cfg = risk_cfg.get("consecutive", {})
    thresholds_cfg = risk_cfg.get("anomaly_score_thresholds", {})

    attn_threshold = consecutive_cfg.get("attention", 1)
    warn_threshold = consecutive_cfg.get("warning", 3)
    severe_threshold = consecutive_cfg.get("severe", 5)

    # 从配置读取异常分数阈值
    sustained_avg_threshold = thresholds_cfg.get("sustained_avg", 1.0)
    high_spike_threshold = thresholds_cfg.get("high_spike", 1.5)

    # 1. 加载最近推理结果
    if daily_results is None:
        daily_results = load_daily_results(elder_id, n_days=7)

    if not daily_results:
        return {
            "elder_id": elder_id,
            "risk_level": 0,
            "risk_label": "正常",
            "risk_types": [],
            "consecutive_deviation": 0,
            "avg_anomaly_7d": 0.0,
            "max_anomaly_7d": 0.0,
            "recommendation": "数据不足，无法判定",
        }

    # 2. 统计连续超标天数
    consecutive_deviation = 0
    for day in reversed(daily_results[-7:]):
        if day.get("is_deviation", False):
            consecutive_deviation += 1
        else:
            break

    # 3. 计算7天聚合分
    recent_scores = [
        d.get("anomaly_score", 0.0) for d in daily_results[-7:]
    ]
    avg_anomaly = float(np.mean(recent_scores))
    max_anomaly = float(np.max(recent_scores))

    # 4. 判定等级
    # 严重级会触发社区网格员介入 + 强提醒，代价高，必须比"提醒"级更严格，
    # 至少要满足同样的幅度门槛（avg_anomaly > sustained_avg）。否则长达数天、
    # 但每天仅"擦线"越过动态阈值的低幅度偏离，会仅凭连续天数直接升到最高级，
    # 与"提醒"级带幅度门槛的判定不一致，且过度打扰家人和社区。
    sustained = avg_anomaly > sustained_avg_threshold
    if consecutive_deviation >= severe_threshold and sustained:
        risk_level = 3
    elif consecutive_deviation >= warn_threshold and sustained:
        risk_level = 2
    elif consecutive_deviation >= attn_threshold or max_anomaly > high_spike_threshold:
        risk_level = 1
    else:
        risk_level = 0

    risk_labels = {0: "正常", 1: "关注", 2: "提醒", 3: "严重"}

    # 5. 分类风险类型
    active_risk_types = []
    try:
        residual_stats = load_residual_stats(elder_id)

        # 获取最新的特征残差
        latest_result = daily_results[-1] if daily_results else {}
        feature_residuals = latest_result.get("feature_residuals", {})

        if feature_residuals and residual_stats is not None:
            risk_type_results = classify_risk_type(
                feature_residuals=feature_residuals,
                residual_stats=residual_stats,
                daily_results=daily_results,
            )
            active_risk_types = [r for r in risk_type_results if r.get("is_active")]
    except Exception as e:
        logger.debug(f"  └─ 风险类型分类跳过: {e}")

    # 6. 生成建议
    recommendation = _generate_recommendation(
        risk_level,
        active_risk_types,
        consecutive_deviation,
    )

    logger.info(
        f"风险判定: elder_id={elder_id}, level={risk_level}({risk_labels[risk_level]}), "
        f"consecutive={consecutive_deviation}, avg_score={avg_anomaly:.4f}"
    )

    return {
        "elder_id": elder_id,
        "risk_level": risk_level,
        "risk_label": risk_labels[risk_level],
        "risk_types": active_risk_types,
        "consecutive_deviation": consecutive_deviation,
        "avg_anomaly_7d": round(avg_anomaly, 4),
        "max_anomaly_7d": round(max_anomaly, 4),
        "recommendation": recommendation,
    }


def _generate_recommendation(
    risk_level: int,
    active_risk_types: list[dict],
    consecutive_deviation: int,
) -> str:
    """根据风险等级和类型生成处置建议"""
    if risk_level == 0:
        return "老人状态稳定，无异常检测"

    if risk_level == 1:
        risk_names = [r["risk_type"] for r in active_risk_types]
        if risk_names:
            return f"轻度关注：{', '.join(risk_names)}指标出现波动，建议持续观察"
        return f"单日轻微偏离（连续{consecutive_deviation}天），建议关注后续变化"

    if risk_level == 2:
        risk_names = [r["risk_type"] for r in active_risk_types]
        names_str = "、".join(risk_names) if risk_names else "多项指标"
        return (
            f"需要提醒：{names_str}已连续{consecutive_deviation}天偏离个人基线，"
            f"建议子女主动联系老人，了解近期生活状态"
        )

    # risk_level == 3
    if active_risk_types:
        risk_names = [r["risk_type"] for r in active_risk_types]
        return (
            f"严重警告：{'、'.join(risk_names)}已连续{consecutive_deviation}天严重偏离基线，"
            f"建议安排上门探访或就医咨询"
        )

    return f"连续{consecutive_deviation}天异常，建议尽快联系老人"


def quick_judge(elder_id: str, today_date: str) -> dict:
    """
    快速判定：加载最新推理结果后直接判定。
    适用于每日调度任务。

    Args:
        elder_id: 老人ID
        today_date: 今日日期

    Returns:
        风险判定结果字典
    """
    daily_results = load_daily_results(elder_id, n_days=7)
    return judge_risk_level(elder_id, daily_results)

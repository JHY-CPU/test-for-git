"""
风险类型判定规则

三种风险类型（全部基于趋势检测）：
    - 抑郁风险：sad_ratio↑ + avg_speed↓ + pitch_variability↓ + distress_events↑
    - 睡眠问题：sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓
    - 社交孤独：social_turns↓ + daily_activity↓ + sad_ratio↑

每种类型有独立的特征贡献权重和阈值。
所有风险均基于连续趋势判定，无单点触发。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskRule:
    """单条风险判定规则"""
    name: str                      # 风险类型名称
    features: list[str]            # 相关特征名（按顺序对应direction和weight）
    # directions 记录每个特征的"异常方向"而非只看绝对偏离。抑郁的语速是"变慢"才异常，
    # 变快不算；睡眠效率是"下降"才异常。只看 |残差| 大会把方向相反的正常波动也误判成风险，
    # 所以必须按方向匹配（见 classify_risk_type 里的 up/down 分支）。
    directions: list[str]          # 异常方向: "up" / "down" / "any"
    weights: list[float]           # 特征在风险评分中的权重
    threshold_ratio: float = 2.0   # 残差超标倍数阈值（对"残差的残差"而非原始残差，见下方二次标准化）
    consecutive_days: int = 3      # 连续超标天数阈值

    def __post_init__(self):
        if len(self.features) != len(self.directions):
            raise ValueError("features and directions must have same length")
        if len(self.features) != len(self.weights):
            raise ValueError("features and weights must have same length")


# ===== 预定义规则 =====

def _load_risk_rules() -> dict[str, RiskRule]:
    """从配置文件加载风险规则权重"""
    from src.utils.io import load_feature_weights

    weights = load_feature_weights()

    # 三类风险的 threshold_ratio / consecutive_days 刻意不同：
    #   - 抑郁：语音信号（SER 在老年嗓音上泛化性存疑）测量噪声较大，threshold_ratio 取 2.0 更严，
    #     避免单路语音抖动就报；3 天连续确认。
    #   - 睡眠：雷达生理指标相对稳定可信，门槛放到 1.5 即可捕捉，仍要 3 天连续。
    #   - 社交孤独：本就是缓变过程（偶尔一两天少说话很正常），要 5 天连续才算趋势，
    #     否则会把老人正常的"安静日"误报成孤独。
    return {
        "depression": RiskRule(
            name="抑郁风险",
            features=["sad_ratio", "avg_speed", "pitch_variability", "distress_events"],
            directions=["up", "down", "down", "up"],
            weights=[
                weights["sad_ratio"],
                weights["avg_speed"],
                weights["pitch_variability"],
                weights["distress_events"],
            ],
            threshold_ratio=2.0,
            consecutive_days=3,
        ),
        "sleep_problem": RiskRule(
            name="睡眠问题",
            features=["sleep_efficiency", "deep_sleep_ratio", "sfi", "hrv_rmssd"],
            directions=["down", "down", "up", "down"],
            weights=[
                weights["sleep_efficiency"],
                weights["deep_sleep_ratio"],
                weights["sfi"],
                weights["hrv_rmssd"],
            ],
            threshold_ratio=1.5,
            consecutive_days=3,
        ),
        "social_isolation": RiskRule(
            name="社交孤独",
            features=["social_turns", "daily_activity", "sad_ratio"],
            directions=["down", "down", "up"],
            weights=[
                weights["social_turns"],
                weights["daily_activity"],
                weights["sad_ratio"],
            ],
            threshold_ratio=1.5,
            consecutive_days=5,
        ),
    }

RISK_RULES = _load_risk_rules()


def classify_risk_type(
    feature_residuals: dict[str, float],
    residual_stats: dict[str, np.ndarray],
    consecutive_days: dict[str, int] | None = None,
    daily_results: list[dict] | None = None,
    today_is_deviation: bool = True,
) -> list[dict]:
    """
    根据当日特征残差判断风险类型。

    Args:
        feature_residuals: {feature_name: residual_value} 当日各特征的标准化残差
        residual_stats: {"mean": np.ndarray(10,), "std": np.ndarray(10,)}
        consecutive_days: 各特征连续异常天数（可选）
        daily_results: 近7天推理结果（用于统计连续天数）

    Returns:
        [
            {
                "risk_type": "抑郁风险",
                "risk_key": "depression",
                "score": 2.3,
                "is_active": True,
                "exceeding_features": ["sad_ratio", "avg_speed"],
                "consecutive_days": 3,
            },
            ...
        ]
    """
    from src.baseline.scaler_utils import FEATURE_NAMES

    if consecutive_days is None:
        consecutive_days = {}

    # 只需 std 作为归一尺度（见下方 normalized_residual 说明，不再用 mean 以免破坏符号）
    if "std" in residual_stats and isinstance(residual_stats["std"], np.ndarray):
        residual_std = {
            FEATURE_NAMES[i]: float(residual_stats["std"][i])
            for i in range(len(FEATURE_NAMES))
        }
    else:
        residual_std = residual_stats.get("std", {})

    results = []

    for risk_key, rule in RISK_RULES.items():
        exceeding_features = []
        total_score = 0.0
        weight_sum = 0.0

        for feat, direction, weight in zip(rule.features, rule.directions, rule.weights):
            feat_value = feature_residuals.get(feat, 0.0)
            feat_std = residual_std.get(feat, 1.0)

            if feat_std < 1e-8:
                feat_std = 1e-8  # 防除零：某特征训练残差几乎恒定时兜底

            # 二次标准化：feature_residuals 是带符号的 GRU 预测残差（actual-pred），
            # 用"训练期残差尺度(std)"归一，得到带符号的 z 分——幅度表示偏离大小，
            # 符号表示方向（正=偏高，负=偏低）。
            # 注意：绝不能减去残差均值。训练残差统计按 |残差| 计（均值恒为正），
            # 若从带符号残差里减这个正均值，会把符号整体拉偏，导致 down 方向永远误触发、
            # 正常特征（残差≈0）也被判为 down 超标。只除以 std（尺度、恒正）即可保号。
            normalized_residual = feat_value / feat_std
            threshold = rule.threshold_ratio

            # 带符号方向判定：up 只认正向超标，down 只认负向超标。这是"方向匹配"的落点，
            # 保证"语速变快""睡眠变好"这类反向偏离不会被计入风险特征。
            is_exceeding = False
            if direction == "up" and normalized_residual > threshold:
                is_exceeding = True
            elif direction == "down" and normalized_residual < -threshold:
                is_exceeding = True
            elif direction == "any" and abs(normalized_residual) > threshold:
                is_exceeding = True

            if is_exceeding:
                exceeding_features.append(feat)

            # 评分累加用 |z| 而非带符号值：即便某特征方向"不对"，它的波动幅度仍反映整体不稳定，
            # 计入加权综合分；但它不会进 exceeding_features，故不满足"方向性超标"的激活前提。
            total_score += abs(normalized_residual) * weight
            weight_sum += weight

        final_score = total_score / weight_sum if weight_sum > 0 else 0.0

        # 当天"达标"信号：今天整体已判偏离 且 ≥1 特征方向性超标 且 加权综合分 > 1.0。
        # 刻意不含连续天数——它只回答"今天够不够格"。连续天数另算。
        #
        # today_is_deviation 这道闸是关键：风险"类型"是对"今天整体异常"的细化分类，
        # 若今天整体都不算偏离（anomaly_score 未超动态阈值），就不该分出任何类型。
        # 否则类型判定用的是固定门槛 final_score>1.0，不受动态阈值保护——正常日
        # |z| 本就常在 1 附近，会频繁误 qualifies、攒够连续天数后把三类都误激活
        # （范围2 诊断：纯高斯正常数据也全线误报）。挂靠整体偏离后，正常日直接不分类。
        qualifies_today = (
            today_is_deviation
            and len(exceeding_features) >= 1
            and final_score > 1.0
        )

        if daily_results is not None:
            cons_days = _count_consecutive_risk_type(
                risk_key, daily_results, qualifies_today
            )
        else:
            cons_days = consecutive_days.get(risk_key, 0) + (1 if qualifies_today else 0)

        # 激活 = 今天达标 且 连续达标天数 ≥ 门槛（3/3/5 天）。这仍是"克制预警"的闸：
        # 方向匹配(qualifies 里) + 幅度门槛(qualifies 里) + 连续趋势(cons_days)。
        is_active = qualifies_today and cons_days >= rule.consecutive_days

        results.append({
            "risk_type": rule.name,
            "risk_key": risk_key,
            "score": round(final_score, 4),
            "is_active": is_active,
            "qualifies": qualifies_today,   # 供 quick_judge 写回日志，次日统计连续天数
            "exceeding_features": exceeding_features,
            "consecutive_days": cons_days,
            "threshold_required": rule.consecutive_days,
        })

    return results


def _day_qualifies(day_result: dict, risk_key: str) -> bool:
    """某历史日志里，该风险类型当天是否"达标"（方向+幅度，不含连续天数）。

    权威来源是 quick_judge 写回的 `risk_type_qualifies` 字典。为兼容旧日志，
    退而读 `risk_types` 里该类型的 qualifies/is_active 标志。
    """
    quals = day_result.get("risk_type_qualifies")
    if isinstance(quals, dict) and risk_key in quals:
        return bool(quals[risk_key])
    # 兼容旧格式：从 risk_types 列表里找
    for rt in day_result.get("risk_types", []) or []:
        if isinstance(rt, dict) and rt.get("risk_key") == risk_key:
            return bool(rt.get("qualifies", rt.get("is_active", False)))
        if rt == risk_key:
            return True
    return False


def _count_consecutive_risk_type(
    risk_key: str,
    daily_results: list[dict],
    today_qualifies: bool | None = None,
) -> int:
    """统计某风险类型"截至今天"的连续达标天数。

    关键修复：今天的达标信号用**现算的** today_qualifies（今天的日志此刻还没写回
    risk_type_qualifies，读日志会漏掉今天，导致永远数不到自己 → 死循环）。
    历史天数从日志的 risk_type_qualifies 读。中间断过即停（真正的"连续"）。

    Args:
        today_qualifies: 今天是否达标（现算）。None 时退化为纯读日志（如周报回溯场景）。
    """
    results = daily_results
    count = 0

    # 今天：用现算值。若今天就不达标，连续数直接 0。
    if today_qualifies is not None:
        if not today_qualifies:
            return 0
        count = 1
        results = daily_results[:-1] if daily_results else []  # 今天那条已用现算值，往前数

    # 历史：从近到远，遇到第一个"不达标"就停
    for day_result in reversed(results):
        if _day_qualifies(day_result, risk_key):
            count += 1
        else:
            break
    return count


def get_risk_feature_importance(risk_key: str) -> dict[str, float]:
    """获取某个风险类型的特征重要性"""
    rule = RISK_RULES.get(risk_key)
    if rule is None:
        return {}
    total = sum(rule.weights)
    return {
        feat: weight / total
        for feat, weight in zip(rule.features, rule.weights)
    }


def list_risk_types() -> list[str]:
    """列出所有风险类型名称"""
    return [rule.name for rule in RISK_RULES.values()]

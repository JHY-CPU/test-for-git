"""
风险类型判定规则

三种风险类型（全部基于趋势检测）：
    - 抑郁风险：sad_ratio↑ + avg_speed↓ + avg_pitch↓ + distress_events↑
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
    directions: list[str]          # 异常方向: "up" / "down" / "any"
    weights: list[float]           # 特征在风险评分中的权重
    threshold_ratio: float = 2.0   # 残差超标倍数阈值
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

    return {
        "depression": RiskRule(
            name="抑郁风险",
            features=["sad_ratio", "avg_speed", "avg_pitch", "distress_events"],
            directions=["up", "down", "down", "up"],
            weights=[
                weights["sad_ratio"],
                weights["avg_speed"],
                weights["avg_pitch"],
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
) -> list[dict]:
    """
    根据当日特征残差判断风险类型。

    Args:
        feature_residuals: {feature_name: residual_value} 当日各特征的标准化残差
        residual_stats: {"mean": np.ndarray(12,), "std": np.ndarray(12,)}
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
    from src.baseline.scaler_utils import FULL_FEATURE_NAMES

    if consecutive_days is None:
        consecutive_days = {}

    if "mean" in residual_stats and isinstance(residual_stats["mean"], np.ndarray):
        residual_mean = {
            FULL_FEATURE_NAMES[i]: float(residual_stats["mean"][i])
            for i in range(len(FULL_FEATURE_NAMES))
        }
        residual_std = {
            FULL_FEATURE_NAMES[i]: float(residual_stats["std"][i])
            for i in range(len(FULL_FEATURE_NAMES))
        }
    else:
        residual_mean = residual_stats.get("mean", {})
        residual_std = residual_stats.get("std", {})

    results = []

    for risk_key, rule in RISK_RULES.items():
        exceeding_features = []
        total_score = 0.0
        weight_sum = 0.0

        for feat, direction, weight in zip(rule.features, rule.directions, rule.weights):
            feat_value = feature_residuals.get(feat, 0.0)
            feat_mean = residual_mean.get(feat, 0.0)
            feat_std = residual_std.get(feat, 1.0)

            if feat_std < 1e-8:
                feat_std = 1e-8

            normalized_residual = (feat_value - feat_mean) / feat_std
            threshold = rule.threshold_ratio

            is_exceeding = False
            if direction == "up" and normalized_residual > threshold:
                is_exceeding = True
            elif direction == "down" and normalized_residual < -threshold:
                is_exceeding = True
            elif direction == "any" and abs(normalized_residual) > threshold:
                is_exceeding = True

            if is_exceeding:
                exceeding_features.append(feat)

            total_score += abs(normalized_residual) * weight
            weight_sum += weight

        final_score = total_score / weight_sum if weight_sum > 0 else 0.0

        if daily_results is not None:
            cons_days = _count_consecutive_risk_type(risk_key, daily_results)
        else:
            cons_days = consecutive_days.get(risk_key, 0)

        is_active = (
            len(exceeding_features) >= 1
            and final_score > 1.0
            and cons_days >= rule.consecutive_days
        )

        results.append({
            "risk_type": rule.name,
            "risk_key": risk_key,
            "score": round(final_score, 4),
            "is_active": is_active,
            "exceeding_features": exceeding_features,
            "consecutive_days": cons_days,
            "threshold_required": rule.consecutive_days,
        })

    return results


def _count_consecutive_risk_type(
    risk_key: str,
    daily_results: list[dict],
) -> int:
    """统计某个风险类型连续活跃的天数"""
    count = 0
    for day_result in reversed(daily_results):
        risk_types = day_result.get("risk_types", [])
        if isinstance(risk_types, list):
            for rt in risk_types:
                if isinstance(rt, dict) and rt.get("risk_key") == risk_key:
                    if rt.get("is_active"):
                        count += 1
                        break
                elif rt == risk_key:
                    count += 1
                    break
            else:
                break
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

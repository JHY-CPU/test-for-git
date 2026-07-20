"""
指标统计模块：误报率、召回率、提前预警天数等
"""

from typing import Any

import numpy as np


def compute_false_positive_rate(
    actual_events: list[bool],
    predicted_events: list[bool],
) -> float:
    """
    计算误报率（False Positive Rate）。

    FPR = FP / (FP + TN) = 假预警次数 / 所有无真实事件的天数

    Args:
        actual_events: 真实事件列表（True=有心理健康事件）
        predicted_events: 系统预警列表（True=系统判定为异常）

    Returns:
        FPR值 [0, 1]
    """
    if len(actual_events) != len(predicted_events):
        raise ValueError("Length mismatch")

    false_positives = 0
    true_negatives = 0

    for actual, predicted in zip(actual_events, predicted_events):
        if not actual and predicted:
            false_positives += 1
        elif not actual and not predicted:
            true_negatives += 1

    total_negatives = false_positives + true_negatives
    if total_negatives == 0:
        return 0.0

    return false_positives / total_negatives


def compute_detection_metrics(
    actual_events: list[bool],
    predicted_events: list[bool],
) -> dict[str, float]:
    """
    计算完整的检测指标。

    Returns:
        {
            "accuracy": 准确率,
            "precision": 精确率,
            "recall": 召回率,
            "f1_score": F1分数,
            "fpr": 误报率,
            "specificity": 特异度,
        }
    """
    if len(actual_events) != len(predicted_events):
        raise ValueError(
            f"Length mismatch: {len(actual_events)} vs {len(predicted_events)}"
        )

    tp = fp = tn = fn = 0
    for actual, predicted in zip(actual_events, predicted_events):
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "specificity": round(specificity, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def compute_early_warning_days(
    actual_event_dates: list[str],
    predicted_event_dates: list[str],
) -> float:
    """
    计算平均提前预警天数。

    系统首次预警日期到真实事件爆发日期的平均天数。
    正值表示提前预警，负值表示滞后。

    Args:
        actual_event_dates: 真实事件爆发日期列表
        predicted_event_dates: 系统首次预警日期列表

    Returns:
        平均提前天数
    """
    if len(actual_event_dates) != len(predicted_event_dates):
        raise ValueError("Length mismatch")

    from datetime import datetime

    lead_days = []
    fmt = "%Y-%m-%d"

    for a_date, p_date in zip(actual_event_dates, predicted_event_dates):
        actual_dt = datetime.strptime(a_date, fmt)
        predicted_dt = datetime.strptime(p_date, fmt)
        lead_days.append((actual_dt - predicted_dt).days)

    return float(np.mean(lead_days))


def compute_daily_alert_rate(
    elder_ids: list[str],
    date_range: tuple[str, str],
    alert_level: int = 1,
) -> dict[str, float]:
    """
    统计每户每天的预警频次。

    Args:
        elder_ids: 老人ID列表
        date_range: (start_date, end_date)
        alert_level: 统计的预警等级下限（≥此等级的才计数）

    Returns:
        {"alerts_per_elder_day": 0.5, "total_alerts": 10, "total_elder_days": 20}
    """
    from datetime import datetime, timedelta

    start, end = date_range
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    total_alerts = 0
    total_elder_days = len(elder_ids) * total_days

    for elder_id in elder_ids:
        # 尝试加载该老人的推理日志
        from src.utils.io import load_daily_results
        try:
            results = load_daily_results(elder_id, n_days=total_days)
            alerts = [
                r for r in results
                if r.get("risk_level", 0) >= alert_level
                and start <= r["date"] <= end
            ]
            total_alerts += len(alerts)
        except Exception:
            pass

    return {
        "alerts_per_elder_day": round(total_alerts / total_elder_days, 4) if total_elder_days > 0 else 0.0,
        "total_alerts": total_alerts,
        "total_elder_days": total_elder_days,
    }


def compare_thresholds(
    personal_results: list[float],
    population_results: list[float],
    ground_truth: list[bool],
) -> dict[str, Any]:
    """
    消融实验：个人基线 vs 群体固定阈值 对比。

    Args:
        personal_results: 个人基线的异常分列表
        population_results: 群体固定阈值的异常标记列表
        ground_truth: 真实事件标记

    Returns:
        对比指标字典
    """
    personal_binary = [s > 1.0 for s in personal_results]
    population_binary = [s > 1.0 for s in population_results]

    personal_metrics = compute_detection_metrics(ground_truth, personal_binary)
    population_metrics = compute_detection_metrics(ground_truth, population_binary)

    return {
        "personal_baseline": personal_metrics,
        "population_threshold": population_metrics,
        "fpr_reduction": round(
            population_metrics["fpr"] - personal_metrics["fpr"], 4
        ),
        "fpr_reduction_ratio": round(
            population_metrics["fpr"] / personal_metrics["fpr"], 2
        ) if personal_metrics["fpr"] > 0 else float("inf"),
    }

"""
指标统计单元测试
"""

import pytest

from src.utils.metrics import (
    compute_false_positive_rate,
    compute_detection_metrics,
    compute_early_warning_days,
    compute_daily_alert_rate,
    compare_thresholds,
)


class TestFPR:
    """测试误报率计算"""

    def test_perfect(self):
        actual = [False, False, False]
        predicted = [False, False, False]
        assert compute_false_positive_rate(actual, predicted) == 0.0

    def test_all_false_positives(self):
        actual = [False, False, False]
        predicted = [True, True, True]
        assert compute_false_positive_rate(actual, predicted) == 1.0

    def test_mixed(self):
        actual = [False, False, False, False]
        predicted = [True, False, True, False]
        assert compute_false_positive_rate(actual, predicted) == 0.5

    def test_no_negatives(self):
        """没有负样本时返回0"""
        actual = [True, True, True]
        predicted = [False, False, False]
        assert compute_false_positive_rate(actual, predicted) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_false_positive_rate([True], [True, False])


class TestDetectionMetrics:
    """测试完整检测指标"""

    def test_perfect_detection(self):
        actual = [True, True, False, False]
        predicted = [True, True, False, False]
        metrics = compute_detection_metrics(actual, predicted)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["fpr"] == 0.0

    def test_all_positive(self):
        actual = [True, False, True, False]
        predicted = [True, True, True, True]
        metrics = compute_detection_metrics(actual, predicted)
        assert metrics["recall"] == 1.0
        assert metrics["fpr"] == 1.0

    def test_all_negative(self):
        actual = [True, True, False, False]
        predicted = [False, False, False, False]
        metrics = compute_detection_metrics(actual, predicted)
        assert metrics["recall"] == 0.0
        assert metrics["fpr"] == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_detection_metrics([True], [True, False])


class TestEarlyWarningDays:
    """测试提前预警天数"""

    def test_early_warning(self):
        actual_dates = ["2026-08-15"]
        predicted_dates = ["2026-08-13"]
        days = compute_early_warning_days(actual_dates, predicted_dates)
        assert days == 2.0

    def test_late_warning(self):
        actual_dates = ["2026-08-13"]
        predicted_dates = ["2026-08-15"]
        days = compute_early_warning_days(actual_dates, predicted_dates)
        assert days == -2.0

    def test_same_day(self):
        actual_dates = ["2026-08-15"]
        predicted_dates = ["2026-08-15"]
        days = compute_early_warning_days(actual_dates, predicted_dates)
        assert days == 0.0


class TestThresholdComparison:
    """测试阈值消融实验"""

    def test_personal_better(self):
        # 个人基线更准确
        personal = [0.5, 0.6, 1.5, 0.4, 0.3]  # 只有1天超标
        population = [1.5, 1.6, 2.0, 1.4, 1.3]  # 每天超标
        ground_truth = [False, False, True, False, False]

        result = compare_thresholds(personal, population, ground_truth)
        assert result["personal_baseline"]["fpr"] < result["population_threshold"]["fpr"]

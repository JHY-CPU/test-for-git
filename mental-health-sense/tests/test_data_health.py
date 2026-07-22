"""
训练数据健康门禁单元测试
"""

import numpy as np
import pytest

from src.baseline.data_health import detect_outlier_days, describe_outlier_days


def _make_normal_block(n_days: int = 14, n_features: int = 10, seed: int = 0) -> np.ndarray:
    """生成一段平稳的正常数据（各特征小幅波动）"""
    rng = np.random.RandomState(seed)
    base = np.array([0.05, 4.5, 32, 0.1, 0.88, 0.30, 5.0, 50, 6000, 35])[:n_features]
    scale = np.array([0.01, 0.2, 2, 0.05, 0.02, 0.02, 0.5, 3, 300, 3])[:n_features]
    return base + rng.normal(0, 1, (n_days, n_features)) * scale


class TestDetectOutlierDays:
    def test_clean_data_no_outliers(self):
        """平稳正常数据不应误判离群天"""
        data = _make_normal_block(seed=1)
        report = detect_outlier_days(data)
        assert report["outlier_day_indices"] == []
        assert report["outlier_ratio"] == 0.0
        assert report["n_days"] == 14

    def test_injected_outlier_day_detected(self):
        """注入一个多特征齐飞的异常天应被识别"""
        data = _make_normal_block(seed=2)
        # 第 5 天：多个特征大幅偏离
        data[5, 0] = 0.9    # sad_ratio 飙升
        data[5, 1] = 1.0    # avg_speed 骤降
        data[5, 3] = 10.0   # distress_events 飙升
        report = detect_outlier_days(data, z_threshold=3.5, min_bad_features=2)
        assert 5 in report["outlier_day_indices"]

    def test_single_feature_spike_below_min_bad(self):
        """只有单个特征离群、未达 min_bad_features 时不判整天离群"""
        data = _make_normal_block(seed=3)
        data[7, 3] = 50.0  # 仅 distress_events 一个特征异常
        report = detect_outlier_days(data, z_threshold=3.5, min_bad_features=2)
        assert 7 not in report["outlier_day_indices"]

    def test_constant_feature_no_divide_by_zero(self):
        """恒定特征不应触发除零或误报"""
        data = _make_normal_block(seed=4)
        data[:, 2] = 30.0  # pitch_variability 完全恒定
        report = detect_outlier_days(data)
        # 不抛异常即可，恒定列不产生离群
        assert not report["feature_flags"][:, 2].any()

    def test_describe_outputs_feature_names(self):
        """描述函数应列出离群特征名"""
        data = _make_normal_block(seed=5)
        data[3, 0] = 0.95
        data[3, 1] = 0.8
        data[3, 3] = 12.0
        report = detect_outlier_days(data, min_bad_features=2)
        lines = describe_outlier_days(data, report)
        assert any("Day#3" in ln for ln in lines)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            detect_outlier_days(np.zeros(10))


class TestColdStartFallback:
    def test_normal_today_no_deviation(self):
        from src.baseline.cold_start_fallback import fallback_deviation_check
        history = _make_normal_block(n_days=10, seed=6)
        today = history.mean(axis=0)  # 完全贴合历史均值
        weights = np.ones(10)
        out = fallback_deviation_check(history, today, weights, sigma=3.0)
        assert out["is_deviation"] is False
        assert out["method"] == "cold_start_fallback"

    def test_extreme_today_flagged(self):
        from src.baseline.cold_start_fallback import fallback_deviation_check
        history = _make_normal_block(n_days=10, seed=7)
        today = history.mean(axis=0).copy()
        today[0] += 20 * history[:, 0].std()  # sad_ratio 远超历史
        today[3] += 20 * history[:, 3].std()
        weights = np.ones(10)
        out = fallback_deviation_check(history, today, weights, sigma=3.0)
        assert out["is_deviation"] is True

    def test_insufficient_history_feature_skipped(self):
        """历史样本 < 2 的特征不参与判定，不应崩溃"""
        from src.baseline.cold_start_fallback import fallback_deviation_check
        history = _make_normal_block(n_days=1, seed=8)
        today = history[0]
        weights = np.ones(10)
        out = fallback_deviation_check(history, today, weights, sigma=3.0)
        assert out["anomaly_score"] == 0.0

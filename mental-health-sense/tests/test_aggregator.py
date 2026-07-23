"""
数据聚合器单元测试
"""

import numpy as np
import pytest

from src.data_pipeline.aggregator import (
    DataInsufficientError,
    aggregate_daily_features,
    aggregate_sleep_features,
    aggregate_activity_features,
    aggregate_social_features,
    aggregate_acoustic_features,
)


class TestSubAggregators:
    """测试子维度聚合"""

    def test_sleep_all_present(self):
        data = {
            "sleep_efficiency": 0.85,
            "deep_sleep_ratio": 0.3,
            "sfi": 5.2,
            "hrv_rmssd": 45.0,
        }
        result = aggregate_sleep_features(data)
        assert result["sleep_efficiency"] == 0.85
        assert result["deep_sleep_ratio"] == 0.3
        assert result["sfi"] == 5.2
        assert result["hrv_rmssd"] == 45.0

    def test_sleep_partial(self):
        data = {"sleep_efficiency": 0.85}
        result = aggregate_sleep_features(data)
        assert result["sleep_efficiency"] == 0.85
        assert result["deep_sleep_ratio"] is None

    def test_sleep_none(self):
        result = aggregate_sleep_features(None)
        assert all(v is None for v in result.values())

    def test_activity_none(self):
        result = aggregate_activity_features(None)
        assert all(v is None for v in result.values())

    def test_social_none(self):
        result = aggregate_social_features(None)
        assert all(v is None for v in result.values())

    def test_acoustic_none(self):
        result = aggregate_acoustic_features(None)
        assert all(v is None for v in result.values())


class TestAggregateDaily:
    """测试每日聚合（特征向量现为10维，已移除时间编码）"""

    def test_all_data_present(self):
        """所有数据维度都完整"""
        vec = aggregate_daily_features(
            date_str="2026-08-01",
            sleep_data={
                "sleep_efficiency": 0.85,
                "deep_sleep_ratio": 0.3,
                "sfi": 5.2,
                "hrv_rmssd": 45.0,
            },
            activity_data={
                "daily_activity": 5000,
            },
            social_data={
                "social_turns": 30,
            },
            acoustic_data={
                "sad_ratio": 0.05,
                "avg_speed": 4.2,
                "pitch_variability": 30.0,
                "distress_events": 0,
            },
        )

        assert vec.shape == (10,), f"Expected (10,), got {vec.shape}"
        assert not np.any(np.isnan(vec))  # 所有10维无NaN

    def test_no_data(self):
        """无任何数据 → 数据不足异常"""
        with pytest.raises(DataInsufficientError) as exc:
            aggregate_daily_features(date_str="2026-08-01")
        assert exc.value.missing_count >= 3

    def test_partial_data_ok(self):
        """少量缺失（<3个）→ 允许通过"""
        vec = aggregate_daily_features(
            date_str="2026-08-01",
            sleep_data={
                "sleep_efficiency": 0.85,
                "deep_sleep_ratio": 0.3,
                "sfi": 5.2,
                "hrv_rmssd": 45.0,
            },
            activity_data={"daily_activity": 5000},
            social_data={"social_turns": 30},
            acoustic_data={"sad_ratio": 0.05, "avg_speed": 4.2},
            # pitch_variability 和 distress_events 缺失（仅2个<3个）
        )
        assert vec.shape == (10,), f"Expected (10,), got {vec.shape}"
        # 有NaN（缺失的特征）
        missing_count = np.sum(np.isnan(vec))
        assert missing_count < 3

    def test_partial_data_insufficient(self):
        """大量缺失（≥3个）→ 异常"""
        with pytest.raises(DataInsufficientError):
            aggregate_daily_features(
                date_str="2026-08-01",
                sleep_data={"sleep_efficiency": 0.85},
                # activity, social, acoustic 全部缺失
            )

    def test_output_ordering(self):
        """验证输出顺序与FEATURE_NAMES一致"""
        from src.baseline.scaler_utils import FEATURE_NAMES

        vec = aggregate_daily_features(
            date_str="2026-08-01",
            sleep_data={
                "sleep_efficiency": 0.85,
                "deep_sleep_ratio": 0.3,
                "sfi": 5.2,
                "hrv_rmssd": 45.0,
            },
            activity_data={"daily_activity": 5000},
            social_data={"social_turns": 30},
            acoustic_data={
                "sad_ratio": 0.05,
                "avg_speed": 4.2,
                "pitch_variability": 30.0,
                "distress_events": 0,
            },
        )

        # sad_ratio 应在 index 0
        sad_idx = FEATURE_NAMES.index("sad_ratio")
        assert vec[sad_idx] == 0.05

        # sleep_efficiency 应在 index 4
        sleep_idx = FEATURE_NAMES.index("sleep_efficiency")
        assert vec[sleep_idx] == 0.85

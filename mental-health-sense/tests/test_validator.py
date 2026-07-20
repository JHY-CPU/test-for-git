"""
数据校验单元测试
"""

import numpy as np
import pytest

from src.data_pipeline.validator import (
    validate_daily_data,
    is_usable_for_training,
    is_usable_for_inference,
    get_quality_summary,
)


def make_vector(health_values):
    """构造10维向量（已移除时间编码）"""
    vec = np.array(health_values, dtype=np.float64)
    assert vec.shape == (10,)
    return vec


class TestValidateDailyData:
    """测试每日数据校验"""

    def test_valid_data(self):
        vec = make_vector([1.0] * 10)
        quality = validate_daily_data(vec, missing_count=0)
        assert quality == "valid"

    def test_insufficient_missing(self):
        vec = make_vector([1.0] * 10)
        quality = validate_daily_data(vec, missing_count=3)
        assert quality == "insufficient"

    def test_insufficient_more_missing(self):
        vec = make_vector([1.0] * 10)
        quality = validate_daily_data(vec, missing_count=5)
        assert quality == "insufficient"

    def test_offline_after_continuous_insufficient(self):
        vec = make_vector([1.0] * 10)
        recent = ["insufficient", "insufficient", "insufficient"]
        quality = validate_daily_data(vec, missing_count=4, recent_quality=recent)
        assert quality == "offline"

    def test_offline_with_mixed_history(self):
        vec = make_vector([1.0] * 10)
        recent = ["insufficient", "offline", "offline"]
        quality = validate_daily_data(vec, missing_count=3, recent_quality=recent)
        assert quality == "offline"

    def test_not_offline_with_interruption(self):
        vec = make_vector([1.0] * 10)
        recent = ["insufficient", "valid", "insufficient"]
        quality = validate_daily_data(vec, missing_count=3, recent_quality=recent)
        assert quality != "offline"

    def test_extreme_outlier(self):
        vec = make_vector([-999.0] + [1.0] * 9)
        quality = validate_daily_data(vec, missing_count=0)
        assert quality == "insufficient"

    def test_edge_case_exactly_3_missing(self):
        vec = make_vector([1.0] * 10)
        quality = validate_daily_data(vec, missing_count=3)
        assert quality == "insufficient"

    def test_edge_case_2_missing_ok(self):
        vec = make_vector([1.0] * 10)
        quality = validate_daily_data(vec, missing_count=2)
        assert quality == "valid"


class TestUsability:
    """测试数据可用性判断"""

    def test_valid_for_training(self):
        assert is_usable_for_training("valid")

    def test_insufficient_not_for_training(self):
        assert not is_usable_for_training("insufficient")
        assert not is_usable_for_training("offline")

    def test_insufficient_not_for_inference(self):
        assert not is_usable_for_inference("insufficient")


class TestQualitySummary:
    """测试质量统计"""

    def test_all_valid(self):
        summary = get_quality_summary(
            ["valid", "valid", "valid", "valid", "valid"]
        )
        assert summary["valid_days"] == 5
        assert summary["insufficient_days"] == 0
        assert summary["valid_ratio"] == 1.0

    def test_mixed(self):
        summary = get_quality_summary(
            ["valid", "insufficient", "valid", "offline", "insufficient"]
        )
        assert summary["valid_days"] == 2
        assert summary["insufficient_days"] == 2
        assert summary["offline_days"] == 1
        assert summary["total_days"] == 5
        assert summary["valid_ratio"] == 0.4

    def test_empty(self):
        summary = get_quality_summary([])
        assert summary["total_days"] == 0
        assert summary["valid_ratio"] == 0.0

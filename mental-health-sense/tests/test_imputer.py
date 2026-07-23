"""
缺失值处理单元测试
"""

import numpy as np
import pytest

from src.data_pipeline.imputer import impute_missing, check_offline_status


class TestImputer:
    """测试缺失值填充（特征向量现为10维，已移除时间编码）"""

    def make_vector(self, health_values):
        """构造10维特征向量"""
        vec = np.array(health_values, dtype=np.float64)
        assert vec.shape == (10,)
        return vec

    def test_no_missing(self):
        """测试无缺失值情况"""
        current = self.make_vector([1.0] * 10)
        filled, missing_count, missing_names = impute_missing(current)

        assert missing_count == 0
        assert missing_names == []
        np.testing.assert_array_almost_equal(filled, current)

    def test_single_missing_with_prev(self):
        """测试单特征缺失 + 前向填充"""
        current = self.make_vector([1.0, np.nan] + [1.0] * 8)
        prev = self.make_vector([1.0, 2.0] + [1.0] * 8)

        filled, missing_count, missing_names = impute_missing(current, prev)

        assert missing_count == 0
        assert filled[1] == 2.0  # 用昨日值填充

    def test_single_missing_without_prev(self):
        """测试单特征缺失 + 无昨日数据"""
        current = self.make_vector([1.0, np.nan] + [1.0] * 8)

        filled, missing_count, missing_names = impute_missing(current)

        # 无法填充，用0代替，但仍记为缺失
        assert filled[1] == 0.0

    def test_multiple_missing_partial_fill(self):
        """测试多特征缺失 + 部分可填充"""
        current = self.make_vector([np.nan, 1.0, np.nan, np.nan] + [1.0] * 6)
        prev = self.make_vector([2.0, 1.0, 3.0, np.nan] + [1.0] * 6)

        filled, missing_count, missing_names = impute_missing(current, prev)

        assert filled[0] == 2.0
        assert filled[2] == 3.0
        assert filled[3] == 0.0  # prev也是NaN，无法填充

    def test_all_health_features_nan(self):
        """测试全部健康特征缺失（极端情况）"""
        current = self.make_vector([np.nan] * 10)
        prev = self.make_vector(list(range(1, 11)))

        filled, missing_count, missing_names = impute_missing(current, prev)

        # 全部可前向填充
        assert missing_count == 0
        for i in range(10):
            assert filled[i] == i + 1

    def test_time_features_never_missing(self):
        """验证10维向量中不含时间编码（时间编码已移除）"""
        from src.baseline.scaler_utils import FEATURE_DIM, FEATURE_NAMES
        assert FEATURE_DIM == 10
        assert "day_sin" not in FEATURE_NAMES
        assert "day_cos" not in FEATURE_NAMES


class TestOfflineCheck:
    """测试离线状态检测"""

    def test_no_offline(self):
        """测试正常情况：不触发离线"""
        quality = ["valid", "valid", "insufficient", "valid", "valid"]
        assert not check_offline_status(quality, threshold=3)

    def test_offline_detected(self):
        """测试检测到离线"""
        quality = ["valid", "insufficient", "insufficient", "insufficient"]
        assert check_offline_status(quality, threshold=3)

    def test_offline_with_offline_marker(self):
        quality = ["valid", "insufficient", "offline", "offline"]
        assert check_offline_status(quality, threshold=3)

    def test_not_enough_data(self):
        """测试数据不足时不误报"""
        quality = ["insufficient", "insufficient"]
        assert not check_offline_status(quality, threshold=3)

    def test_threshold_custom(self):
        quality = ["insufficient", "insufficient"]
        assert check_offline_status(quality, threshold=2)

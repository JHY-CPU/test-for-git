"""
EWMA累积基线单元测试
"""

import numpy as np
import pytest

from src.baseline.ewma import CumulativeEWMABaseline


class TestCumulativeEWMABaseline:
    """测试EWMA累积基线"""

    def test_initial_state(self):
        """测试初始状态"""
        ewma = CumulativeEWMABaseline(alpha=0.05)
        assert ewma.mean is None
        assert ewma.n == 0
        assert ewma.std == 1.0  # n<2默认返回1.0

    def test_single_update(self):
        """测试单次更新"""
        ewma = CumulativeEWMABaseline(alpha=0.1)
        ewma.update(1.5)
        assert ewma.mean == 1.5
        assert ewma.n == 1
        assert len(ewma.history) == 1

    def test_multiple_updates(self):
        """测试多次更新：EWMA收敛性"""
        ewma = CumulativeEWMABaseline(alpha=0.1)
        values = [1.0, 1.0, 1.0, 1.0, 1.0]
        for v in values:
            ewma.update(v)

        assert abs(ewma.mean - 1.0) < 0.01
        assert ewma.n == 5
        assert len(ewma.history) == 5

    def test_threshold_calculation(self):
        """测试阈值计算"""
        ewma = CumulativeEWMABaseline(alpha=0.1)

        # 插入稳定数据
        for _ in range(30):
            ewma.update(1.0)

        threshold = ewma.get_threshold(2.5)
        assert threshold > ewma.mean

        # 插入异常数据
        ewma.update(10.0)
        threshold_after = ewma.get_threshold(2.5)
        assert threshold_after > threshold

    def test_alpha_validation(self):
        """测试alpha参数校验"""
        with pytest.raises(ValueError, match="alpha must be in"):
            CumulativeEWMABaseline(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            CumulativeEWMABaseline(alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            CumulativeEWMABaseline(alpha=1.5)

    def test_serialization(self):
        """测试序列化/反序列化往返"""
        ewma = CumulativeEWMABaseline(alpha=0.05)
        for v in [1.0, 1.2, 0.8, 1.1, 0.9]:
            ewma.update(v)

        data = ewma.to_dict()
        restored = CumulativeEWMABaseline.from_dict(data)

        assert restored.alpha == ewma.alpha
        assert restored.mean == ewma.mean
        assert restored.n == ewma.n
        assert restored.std == ewma.std
        assert restored.history == ewma.history

    def test_save_load(self, tmp_path):
        """测试文件保存/加载"""
        ewma = CumulativeEWMABaseline(alpha=0.05)
        for v in [1.0, 1.2, 0.8]:
            ewma.update(v)

        filepath = tmp_path / "test_ewma.pkl"
        ewma.save(filepath)

        loaded = CumulativeEWMABaseline.load(filepath)
        assert loaded.mean == ewma.mean
        assert loaded.n == ewma.n

    def test_reset(self):
        """测试重置功能"""
        ewma = CumulativeEWMABaseline(alpha=0.05)
        for v in [1.0, 2.0, 3.0]:
            ewma.update(v)

        ewma.reset()
        assert ewma.mean is None
        assert ewma.n == 0
        assert len(ewma.history) == 0

    def test_percentile(self):
        """测试百分位数计算"""
        ewma = CumulativeEWMABaseline(alpha=0.1)
        for v in range(1, 101):
            ewma.update(float(v))

        p50 = ewma.get_percentile(50)
        assert 45 < p50 < 55

        p95 = ewma.get_percentile(95)
        assert p95 > p50

    def test_invalid_update_type(self):
        """测试非法输入"""
        ewma = CumulativeEWMABaseline()
        with pytest.raises(TypeError):
            ewma.update("not_a_number")

    def test_std_stability(self):
        """测试标准差的数值稳定性"""
        ewma = CumulativeEWMABaseline(alpha=0.05)
        # 大量相同值
        for _ in range(100):
            ewma.update(1.0)
        assert ewma.std < 0.1

        # 加入一个极值后
        ewma.update(100.0)
        assert ewma.std > 1.0

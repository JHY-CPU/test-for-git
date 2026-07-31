"""抑郁评估状态白名单测试

这个文件存在的理由与 src/utils/status.py 一样：本仓在状态白名单上吃过一次大亏
（cold_start_fallback 只被加进 5 处白名单里的 2 处，建档期 35 天完全没有预警能力，
见 VALIDATION §8.2）。抑郁通道从第一天就把白名单钉在测试里。
"""

import pytest

from src.depression.status import (
    ALL_STATUSES,
    DISPLAYABLE_STATUSES,
    STATUS_ASSESSED,
    STATUS_FAILED,
    STATUS_LABELS,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_CLIP,
    STATUS_NO_SOURCE,
    STATUS_STALE,
    is_displayable,
    is_weak_evidence,
    validate_status,
)


class TestDisplayable:
    def test_只有两个状态可展示分数(self):
        assert DISPLAYABLE_STATUSES == {STATUS_ASSESSED, STATUS_LOW_CONFIDENCE}

    def test_low_confidence_可展示(self):
        """出了分但证据弱 ≠ 没有结论。

        藏起来会让家属以为系统没测，而实际上是测了、只是把握不大——
        与 build_mpdd_evidence._quality 对 cold_start 的处理同理：
        证据比 valid 弱，但绝不是 missing。
        """
        assert is_displayable(STATUS_LOW_CONFIDENCE)
        assert is_weak_evidence(STATUS_LOW_CONFIDENCE)

    @pytest.mark.parametrize(
        "status", [STATUS_STALE, STATUS_NO_CLIP, STATUS_NO_SOURCE, STATUS_FAILED]
    )
    def test_无结论状态一律不展示(self, status):
        assert not is_displayable(status)

    def test_未知状态不展示(self):
        assert not is_displayable("something_else")
        assert not is_displayable(None)


class TestValidateStatus:
    def test_合法状态原样返回(self):
        assert validate_status(STATUS_ASSESSED) == STATUS_ASSESSED

    def test_非法状态立即报错(self):
        """打错的状态名若静默通过，展示层会一路当成"不可显示"，
        家属看到"暂无评估"而评估其实是成功的——日志里没有任何痕迹。"""
        with pytest.raises(ValueError, match="Unknown depression status"):
            validate_status("assess")


class TestLabels:
    def test_每个状态都有中文标签(self):
        assert set(STATUS_LABELS) == ALL_STATUSES

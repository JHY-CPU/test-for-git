"""周报统计指标单元测试

★ 本文件是为一个"一直在说谎但没人发现"的缺陷补的回归。

旧实现取 `r.get('anomaly_score', 0)` 计算"平均异常分/最高异常分"——但双轨
改造后 anomaly_score 只存在于 r["sleep"] / r["social"] 里，顶层根本没有这个键
（实测 60/60 个推理日志都没有）。于是这两行**恒为 0.00**，而真实分数是
0.63~1.13。周报把"一切正常"当成事实报了出去，而且不会报错、不会告警。

周报是家属实际会看的唯一产物，这种"静默的正确外观"比崩溃危险得多。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import TRACK_SLEEP, TRACK_SOCIAL
from src.report.weekly_report import _format_track_stats


def day(sleep_score=None, social_score=None, sleep_dev=False, social_dev=False):
    result = {"is_deviation": sleep_dev or social_dev}
    if sleep_score is not None:
        result[TRACK_SLEEP] = {
            "anomaly_score": sleep_score, "is_deviation": sleep_dev,
        }
    if social_score is not None:
        result[TRACK_SOCIAL] = {
            "anomaly_score": social_score, "is_deviation": social_dev,
        }
    return result


class TestFormatTrackStats:
    def test_reports_real_scores_not_zero(self):
        """回归核心：真实分数必须出现在文案里，不能是 0.00"""
        week = [day(sleep_score=s, social_score=0.5) for s in (0.63, 1.04, 1.13)]
        text = _format_track_stats(week)

        assert "0.00" not in text, f"分数不该恒为 0.00：\n{text}"
        assert "1.13" in text, f"最高分应为 1.13：\n{text}"

    def test_per_track_lines(self):
        week = [day(sleep_score=1.0, social_score=2.0) for _ in range(3)]
        text = _format_track_stats(week)
        assert "睡眠轨" in text and "社会连接轨" in text

    def test_tracks_are_not_averaged_together(self):
        """★ 两轨的分绝不能平均——8 维 vs 5 维、权重和不同，尺度不可比。

        与 judge"每轨各自算等级、取较高者"是同一条不变量。
        """
        week = [day(sleep_score=3.0, social_score=0.0) for _ in range(3)]
        text = _format_track_stats(week)

        assert "3.00" in text, "睡眠轨的高分必须原样呈现"
        assert "1.50" not in text, "出现 1.50 说明两轨被平均了"

    def test_deviation_days_counted_per_track(self):
        week = [
            day(sleep_score=2.0, social_score=0.5, sleep_dev=True),
            day(sleep_score=2.0, social_score=0.5, sleep_dev=True),
            day(sleep_score=0.5, social_score=0.5),
        ]
        text = _format_track_stats(week)
        assert "偏离 2/3 天" in text, f"睡眠轨应为 2/3：\n{text}"
        assert "偏离 0/3 天" in text, f"社交轨应为 0/3：\n{text}"

    def test_missing_track_reported_not_silently_zero(self):
        """某轨整周无数据时要明说，不能显示成 0.00 的"正常"。

        "测不到"呈现为"一切正常"是本项目反复防的失效模式。
        """
        week = [day(sleep_score=1.0) for _ in range(3)]
        text = _format_track_stats(week)
        assert "社会连接轨：本周无有效数据" in text

    def test_empty_week_does_not_crash(self):
        assert _format_track_stats([])

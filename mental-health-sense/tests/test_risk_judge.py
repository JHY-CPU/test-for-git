"""
风险等级判定单元测试
"""

import pytest

from src.risk.judge import judge_risk_level


class TestJudgeRiskLevel:
    """测试风险等级判定"""

    def make_result(self, date, anomaly_score, is_deviation, feature_residuals=None):
        """构造每日推理结果"""
        return {
            "elder_id": "E001",
            "date": date,
            "anomaly_score": anomaly_score,
            "is_deviation": is_deviation,
            "feature_residuals": feature_residuals or {},
        }

    def test_no_data(self):
        """无数据时返回正常"""
        result = judge_risk_level("E001", daily_results=[])
        assert result["risk_level"] == 0
        assert result["risk_label"] == "正常"

    def test_normal_single_day(self):
        """正常日判定"""
        results = [
            self.make_result("2026-08-01", 0.5, False),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 0

    def test_single_deviation_attention(self):
        """单日偏离 → 一级关注"""
        results = [
            self.make_result("2026-08-01", 1.2, True),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 1
        assert result["risk_label"] == "关注"

    def test_three_consecutive_warning(self):
        """连续3天偏离 → 二级提醒"""
        results = [
            self.make_result("2026-08-01", 1.2, True),
            self.make_result("2026-08-02", 1.3, True),
            self.make_result("2026-08-03", 1.4, True),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 2
        assert result["risk_label"] == "提醒"

    def test_five_consecutive_severe(self):
        """连续5天偏离 → 三级严重"""
        results = [
            self.make_result(f"2026-08-0{i}", 1.2 + i * 0.1, True)
            for i in range(1, 6)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 3
        assert result["risk_label"] == "严重"

    def test_intermittent_deviation(self):
        """间断偏离（中间有一天正常）"""
        results = [
            self.make_result("2026-08-01", 1.2, True),
            self.make_result("2026-08-02", 0.5, False),  # 正常日中断
            self.make_result("2026-08-03", 1.3, True),
        ]
        result = judge_risk_level("E001", daily_results=results)
        # 连续计数中断，应不超过一级
        assert result["risk_level"] <= 1

    def test_high_single_spike_attention(self):
        """单日极高异常分 → 一级关注"""
        results = [
            self.make_result("2026-08-01", 5.0, True),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] >= 1

    def test_recovery_after_deviation(self):
        """偏离后恢复正常"""
        results = [
            self.make_result("2026-08-01", 1.2, True),
            self.make_result("2026-08-02", 1.3, True),
            self.make_result("2026-08-03", 0.4, False),
            self.make_result("2026-08-04", 0.3, False),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 0

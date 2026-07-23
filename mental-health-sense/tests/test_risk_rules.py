"""
风险判定规则单元测试
"""

import numpy as np
import pytest

from src.risk.rules import (
    classify_risk_type,
    RISK_RULES,
    RiskRule,
    get_risk_feature_importance,
    list_risk_types,
)


class TestRiskRule:
    """测试风险规则定义"""

    def test_valid_rule_creation(self):
        rule = RiskRule(
            name="测试规则",
            features=["a", "b"],
            directions=["up", "down"],
            weights=[1.0, 2.0],
        )
        assert rule.name == "测试规则"
        assert rule.threshold_ratio == 2.0  # 默认值

    def test_rule_validation_length_mismatch(self):
        with pytest.raises(ValueError):
            RiskRule(
                name="坏规则",
                features=["a", "b"],
                directions=["up"],  # 长度不匹配
                weights=[1.0, 2.0],
            )

    def test_all_predefined_rules_valid(self):
        """所有预定义规则应该是合法的"""
        for key, rule in RISK_RULES.items():
            assert len(rule.features) == len(rule.directions)
            assert len(rule.features) == len(rule.weights)
            assert rule.name
            assert rule.consecutive_days >= 1


class TestClassifyRiskType:
    """测试风险类型分类"""

    def make_residual_stats(self):
        """构造残差统计"""
        return {
            "mean": np.zeros(12),
            "std": np.ones(12),
        }

    def test_normal_day_no_risk(self):
        """测试正常日：不触发任何风险"""
        from src.baseline.scaler_utils import FEATURE_NAMES

        # 所有残差都接近0（完全正常）
        feature_residuals = {name: 0.01 for name in FEATURE_NAMES}

        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=self.make_residual_stats(),
        )

        assert all(not r["is_active"] for r in results)

    def test_depression_triggered(self):
        """测试抑郁风险触发"""
        from src.baseline.scaler_utils import FEATURE_NAMES

        feature_residuals = {name: 0.0 for name in FEATURE_NAMES}
        # sad_ratio↑ + avg_speed↓ + pitch_variability↓ + distress_events↑
        feature_residuals["sad_ratio"] = 3.0
        feature_residuals["avg_speed"] = -3.0
        feature_residuals["pitch_variability"] = -2.5
        feature_residuals["distress_events"] = 2.5

        consecutive = {"depression": 3}

        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=self.make_residual_stats(),
            consecutive_days=consecutive,
        )

        dep_result = next(r for r in results if r["risk_key"] == "depression")
        assert dep_result["is_active"]

    def test_sleep_problem_triggered(self):
        """测试睡眠问题触发"""
        from src.baseline.scaler_utils import FEATURE_NAMES

        feature_residuals = {name: 0.0 for name in FEATURE_NAMES}
        # sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓
        feature_residuals["sleep_efficiency"] = -3.0
        feature_residuals["deep_sleep_ratio"] = -3.0
        feature_residuals["sfi"] = 3.0
        feature_residuals["hrv_rmssd"] = -2.5

        consecutive = {"sleep_problem": 3}

        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=self.make_residual_stats(),
            consecutive_days=consecutive,
        )

        sleep_result = next(r for r in results if r["risk_key"] == "sleep_problem")
        assert sleep_result["is_active"]

    def test_feature_importance(self):
        """测试特征重要性获取"""
        importance = get_risk_feature_importance("depression")
        assert len(importance) == 4
        assert "sad_ratio" in importance
        assert "avg_speed" in importance
        assert "pitch_variability" in importance
        assert "distress_events" in importance

        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_risk_key(self):
        """测试不存在的风险类型"""
        importance = get_risk_feature_importance("nonexistent")
        assert importance == {}

    def test_list_risk_types(self):
        """测试列出所有风险类型"""
        types = list_risk_types()
        assert "抑郁风险" in types
        assert "睡眠问题" in types
        assert "社交孤独" in types
        assert len(types) == 3

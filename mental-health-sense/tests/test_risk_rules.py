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
        """构造残差统计（6维）"""
        return {
            "mean": np.zeros(6),
            "std": np.ones(6),
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

    def test_social_isolation_triggered(self):
        """测试社交孤独触发（social_turns↓ + daily_activity↓）"""
        from src.baseline.scaler_utils import FEATURE_NAMES

        feature_residuals = {name: 0.0 for name in FEATURE_NAMES}
        feature_residuals["social_turns"] = -3.0
        feature_residuals["daily_activity"] = -3.0

        consecutive = {"social_isolation": 5}

        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=self.make_residual_stats(),
            consecutive_days=consecutive,
        )

        soc_result = next(r for r in results if r["risk_key"] == "social_isolation")
        assert soc_result["is_active"]

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

    def test_normal_day_positive_abs_mean_stats(self):
        """回归：残差≈0 且 residual_stats 均值为正（abs 残差统计）时不得误报。

        修复前 classify_risk_type 从带符号残差里减去 abs 均值，导致正常特征
        （残差≈0）在 down 方向被判为超标，sleep_problem/social_isolation 直接误活跃。
        """
        from src.baseline.scaler_utils import FEATURE_NAMES

        feature_residuals = {name: 0.0 for name in FEATURE_NAMES}
        stats = {"mean": np.full(6, 0.5), "std": np.full(6, 0.3)}
        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=stats,
            consecutive_days={k: 9 for k in ("sleep_problem", "social_isolation")},
        )
        assert all(not r["is_active"] for r in results)
        assert all(r["exceeding_features"] == [] for r in results)

    def test_down_feature_improving_not_flagged(self):
        """方向性：down 特征"变好"（残差为正）不得计入超标。

        sleep_efficiency 方向为 down（下降才异常）。今日睡眠效率高于预测（residual>0）
        是好事，不应进入 exceeding_features。
        """
        from src.baseline.scaler_utils import FEATURE_NAMES

        feature_residuals = {name: 0.0 for name in FEATURE_NAMES}
        feature_residuals["sleep_efficiency"] = 3.0  # 睡眠效率比预测更高（正残差）→ 非异常
        stats = {"mean": np.zeros(6), "std": np.ones(6)}
        results = classify_risk_type(
            feature_residuals=feature_residuals,
            residual_stats=stats,
            consecutive_days={"sleep_problem": 9},
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_problem")
        assert "sleep_efficiency" not in sleep["exceeding_features"]

    def test_feature_importance(self):
        """测试特征重要性获取"""
        importance = get_risk_feature_importance("sleep_problem")
        assert len(importance) == 4
        assert "sleep_efficiency" in importance
        assert "deep_sleep_ratio" in importance
        assert "sfi" in importance
        assert "hrv_rmssd" in importance

        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_risk_key(self):
        """测试不存在的风险类型"""
        importance = get_risk_feature_importance("nonexistent")
        assert importance == {}

    def test_list_risk_types(self):
        """测试列出所有风险类型"""
        types = list_risk_types()
        assert "睡眠问题" in types
        assert "社交孤独" in types
        assert len(types) == 2

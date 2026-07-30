"""
风险判定规则单元测试（三类，双轨）
"""

import pytest

from src.baseline.scaler_utils import (
    SLEEP_FEATURES,
    SOCIAL_FEATURES,
    TRACK_SLEEP,
    TRACK_SOCIAL,
)
from src.risk.rules import (
    RiskRule,
    build_risk_rules,
    classify_risk_type,
    get_risk_feature_importance,
    list_risk_types,
)


def track_result(track: str, overrides: dict | None = None, status: str = "success") -> dict:
    """构造某轨的推理结果，signed_z 默认全 0（完全正常）"""
    names = SLEEP_FEATURES if track == TRACK_SLEEP else SOCIAL_FEATURES
    signed_z = {n: 0.0 for n in names}
    if overrides:
        signed_z.update(overrides)
    return {
        "track": track,
        "status": status,
        "signed_available": True,
        "signed_z": signed_z,
        "anomaly_score": 1.0,
        "is_deviation": True,
    }


def both_tracks(sleep_z: dict | None = None, social_z: dict | None = None) -> dict:
    return {
        TRACK_SLEEP: track_result(TRACK_SLEEP, sleep_z),
        TRACK_SOCIAL: track_result(TRACK_SOCIAL, social_z),
    }


def history(risk_key: str, n_days: int, quality: str = "valid") -> list[dict]:
    """构造 n 天"该风险类型已达标"的历史日志 + 今天占位一条"""
    days = [
        {"day_key": f"2026-08-{i + 1:02d}", "data_quality": quality,
         "risk_type_qualifies": {risk_key: True}}
        for i in range(n_days)
    ]
    days.append({"day_key": "2026-08-31", "data_quality": quality})  # 今天，尚未写回
    return days


class TestRuleDefinitions:
    def test_three_rules_exist(self):
        rules = build_risk_rules()
        assert set(rules) == {"sleep_stability", "social_decline", "circadian_disruption"}

    def test_rules_wellformed(self):
        for key, rule in build_risk_rules().items():
            assert isinstance(rule, RiskRule)
            assert rule.name
            assert rule.required, f"{key} 必须有必需特征"
            assert rule.consecutive_days >= 1
            for track, feat, direction in rule.all_features():
                assert track in (TRACK_SLEEP, TRACK_SOCIAL)
                assert direction in ("up", "down", "any")

    def test_social_uses_rolling_window(self):
        """社会连接减弱用 7 天滚动窗 ≥5 天，不用严格连续"""
        rule = build_risk_rules()["social_decline"]
        assert rule.uses_rolling()
        assert rule.rolling_window == 7
        assert rule.rolling_required == 5

    def test_social_requires_all_three(self):
        """三项全中：copresence↓ 且 out_of_home↓ 且 activity↓"""
        rule = build_risk_rules()["social_decline"]
        feats = {f for _, f, _ in rule.required}
        assert feats == {"copresence_min", "out_of_home_min", "activity_counts"}
        assert rule.optional == []

    def test_circadian_is_cross_track(self):
        """作息节律紊乱跨轨：RA/IV 在社交轨，sleep_onset_clock 在睡眠轨"""
        rule = build_risk_rules()["circadian_disruption"]
        assert {t for t, _, _ in rule.required} == {TRACK_SOCIAL}
        assert {t for t, _, _ in rule.optional} == {TRACK_SLEEP}
        assert rule.consecutive_days_degraded == 7

    def test_list_risk_types(self):
        types = list_risk_types()
        assert types == ["睡眠稳定性偏离", "社会连接减弱", "作息节律紊乱"]


class TestNormalDay:
    def test_all_zero_no_risk(self):
        results = classify_risk_type(both_tracks(), daily_results=[])
        assert all(not r["is_active"] for r in results)
        assert all(not r["qualifies"] for r in results)
        assert all(r["exceeding_features"] == [] for r in results)

    def test_long_history_still_no_risk_when_today_normal(self):
        """★ 今天不达标 → 连续天数归零，历史再长也不激活"""
        results = classify_risk_type(
            both_tracks(), daily_results=history("sleep_stability", 10)
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert not sleep["is_active"]
        assert sleep["consecutive_days"] == 0


class TestDirectionality:
    def test_down_feature_improving_not_flagged(self):
        """sleep_efficiency 方向为 down；今日高于预测（signed_z>0）是好事，不算超标"""
        results = classify_risk_type(
            both_tracks(sleep_z={"sleep_efficiency": 3.0, "waso_min": 3.0}),
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert "sleep_efficiency" not in sleep["exceeding_features"]
        assert not sleep["qualifies"], "必需项未全中时不应达标"

    def test_up_feature_decreasing_not_flagged(self):
        """waso_min 方向为 up；今日低于预测（睡得更好）不算超标"""
        results = classify_risk_type(
            both_tracks(sleep_z={"sleep_efficiency": -3.0, "waso_min": -3.0}),
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert "waso_min" not in sleep["exceeding_features"]

    def test_any_direction_both_ways(self):
        """sleep_onset_clock 方向为 any：提前或延后都算漂移"""
        for onset in (2.5, -2.5):
            results = classify_risk_type(
                both_tracks(
                    sleep_z={"sleep_onset_clock": onset},
                    social_z={"rar_amplitude": -3.0, "rar_iv": 2.8},
                ),
                daily_results=[],
            )
            circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
            assert "sleep_onset_clock" in circ["exceeding_features"], f"onset={onset}"


class TestSleepStability:
    def test_requires_both_se_and_waso(self):
        """只有 SE↓ 而 WASO 正常 → 不达标（必需项必须全中）"""
        results = classify_risk_type(
            both_tracks(sleep_z={"sleep_efficiency": -4.0, "sol_min": 3.0}),
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert not sleep["qualifies"]

    def test_requires_at_least_one_optional(self):
        """SE↓ 且 WASO↑ 但三个可选项都正常 → 不达标"""
        results = classify_risk_type(
            both_tracks(sleep_z={"sleep_efficiency": -4.0, "waso_min": 4.0}),
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert not sleep["qualifies"]

    def test_triggers_with_required_plus_one_optional(self):
        results = classify_risk_type(
            both_tracks(sleep_z={
                "sleep_efficiency": -4.0, "waso_min": 4.0, "bed_exit_count": 3.0,
            }),
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert sleep["qualifies"]
        assert sleep["is_active"], "连续 5 天 ≥ 门槛 3 天，应激活"
        assert set(sleep["exceeding_features"]) >= {
            "sleep_efficiency", "waso_min", "bed_exit_count"
        }

    def test_not_active_before_threshold(self):
        """连续 2 天 < 门槛 3 天 → 达标但不激活"""
        results = classify_risk_type(
            both_tracks(sleep_z={
                "sleep_efficiency": -4.0, "waso_min": 4.0, "bed_exit_count": 3.0,
            }),
            daily_results=history("sleep_stability", 1),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert sleep["qualifies"]
        assert not sleep["is_active"]
        assert sleep["consecutive_days"] == 2


class TestSocialDecline:
    ALL_DOWN = {"copresence_min": -3.0, "out_of_home_min": -3.0, "activity_counts": -3.0}

    def test_requires_all_three(self):
        """只有两项下降 → 不达标"""
        results = classify_risk_type(
            both_tracks(social_z={"copresence_min": -3.0, "out_of_home_min": -3.0}),
            daily_results=history("social_decline", 6),
        )
        social = next(r for r in results if r["risk_key"] == "social_decline")
        assert not social["qualifies"]

    def test_triggers_with_all_three(self):
        results = classify_risk_type(
            both_tracks(social_z=self.ALL_DOWN),
            daily_results=history("social_decline", 6),
        )
        social = next(r for r in results if r["risk_key"] == "social_decline")
        assert social["qualifies"]
        assert social["is_active"]

    def test_rolling_window_tolerates_gap(self):
        """★ 滚动窗的价值：中间断一天仍能凑满 5/7，严格连续则会归零"""
        days = [
            {"day_key": "2026-08-01", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-02", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-03", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": False}},   # 周末探访，断一天
            {"day_key": "2026-08-04", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-05", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-06", "data_quality": "valid"},   # 今天
        ]
        results = classify_risk_type(both_tracks(social_z=self.ALL_DOWN), daily_results=days)
        social = next(r for r in results if r["risk_key"] == "social_decline")
        assert social["consecutive_days"] == 5, "4 天历史达标 + 今天 = 5"
        assert social["is_active"]

    def test_degraded_days_skipped_not_counted(self):
        """degraded 日既不累加也不打断——传感器抖动不该攒成预警，也不该清零"""
        days = [
            {"day_key": "2026-08-01", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-02", "data_quality": "degraded",
             "risk_type_qualifies": {"social_decline": True}},   # 应被跳过
            {"day_key": "2026-08-03", "data_quality": "valid",
             "risk_type_qualifies": {"social_decline": True}},
            {"day_key": "2026-08-04", "data_quality": "valid"},  # 今天
        ]
        results = classify_risk_type(both_tracks(social_z=self.ALL_DOWN), daily_results=days)
        social = next(r for r in results if r["risk_key"] == "social_decline")
        assert social["consecutive_days"] == 3, "2 个 valid 历史日 + 今天，degraded 日不计入"


class TestCircadianDisruption:
    RA_IV = {"rar_amplitude": -3.1, "rar_iv": 2.8}

    def test_triggers_with_onset_drift(self):
        results = classify_risk_type(
            both_tracks(sleep_z={"sleep_onset_clock": 2.4}, social_z=self.RA_IV),
            daily_results=history("circadian_disruption", 6),
        )
        circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
        assert circ["qualifies"]
        assert circ["is_active"]
        assert circ["threshold_required"] == 5

    def test_no_trigger_without_onset_drift(self):
        """RA↓ IV↑ 但入睡时刻正常 → 可选项未满足，不达标"""
        results = classify_risk_type(
            both_tracks(social_z=self.RA_IV),
            daily_results=history("circadian_disruption", 6),
        )
        circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
        assert not circ["qualifies"]

    def test_degraded_mode_when_sleep_track_offline(self):
        """★ 睡眠轨离线 → 免除 onset 要求，但门槛从 5 天提到 7 天"""
        results = classify_risk_type(
            {
                TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start",
                              "signed_available": False},
                TRACK_SOCIAL: track_result(TRACK_SOCIAL, self.RA_IV),
            },
            daily_results=history("circadian_disruption", 6),
        )
        circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
        assert circ["qualifies"]
        assert circ["degraded_mode"]
        assert circ["threshold_required"] == 7

    def test_degraded_mode_below_threshold_not_active(self):
        """降级模式下连续 5 天（正常门槛）不足以激活——必须攒到 7 天"""
        results = classify_risk_type(
            {
                TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start",
                              "signed_available": False},
                TRACK_SOCIAL: track_result(TRACK_SOCIAL, self.RA_IV),
            },
            daily_results=history("circadian_disruption", 4),
        )
        circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
        assert circ["consecutive_days"] == 5
        assert circ["threshold_required"] == 7
        assert not circ["is_active"], "5 天 < 降级门槛 7 天，不应激活"

    def test_degraded_mode_activates_at_seven(self):
        results = classify_risk_type(
            {
                TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start",
                              "signed_available": False},
                TRACK_SOCIAL: track_result(TRACK_SOCIAL, self.RA_IV),
            },
            daily_results=history("circadian_disruption", 7),
        )
        circ = next(r for r in results if r["risk_key"] == "circadian_disruption")
        assert circ["consecutive_days"] == 8
        assert circ["is_active"]


class TestTrackAvailability:
    def test_missing_required_track_not_evaluable(self):
        """必需轨不可用 → 标记为不可评估，而不是当成『正常』"""
        results = classify_risk_type(
            {
                TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start",
                              "signed_available": False},
                TRACK_SOCIAL: track_result(TRACK_SOCIAL),
            },
            daily_results=[],
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert not sleep["evaluable"]
        assert sleep["skip_reason"] is not None
        assert not sleep["is_active"]

    def test_old_format_stats_blocks_direction_judgment(self):
        """旧格式基线（无 signed 统计）→ 不做方向判定，避免用错量纲给出结论"""
        results = classify_risk_type(
            {
                TRACK_SLEEP: {**track_result(TRACK_SLEEP, {
                    "sleep_efficiency": -4.0, "waso_min": 4.0, "sol_min": 3.0,
                }), "signed_available": False},
                TRACK_SOCIAL: track_result(TRACK_SOCIAL),
            },
            daily_results=history("sleep_stability", 5),
        )
        sleep = next(r for r in results if r["risk_key"] == "sleep_stability")
        assert not sleep["evaluable"]


class TestFeatureImportance:
    def test_sleep_importance_sums_to_one(self):
        imp = get_risk_feature_importance("sleep_stability")
        assert set(imp) == {
            "sleep_efficiency", "waso_min", "sol_min", "bed_exit_count", "deep_sleep_ratio",
        }
        assert abs(sum(imp.values()) - 1.0) < 1e-9

    def test_cross_track_importance(self):
        """跨轨规则的重要性要能同时取到两轨的权重"""
        imp = get_risk_feature_importance("circadian_disruption")
        assert set(imp) == {"rar_amplitude", "rar_iv", "sleep_onset_clock"}
        assert abs(sum(imp.values()) - 1.0) < 1e-9

    def test_unknown_key(self):
        assert get_risk_feature_importance("nonexistent") == {}

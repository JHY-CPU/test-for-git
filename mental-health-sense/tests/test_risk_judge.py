"""
风险等级判定单元测试（双轨）

关键设计：**不跨轨比较绝对分**。两轨的残差尺度不同（8 维 vs 5 维，权重和不同），
把 anomaly_sleep 与 anomaly_social 取最大或求平均都没有统计意义。
等级判定改为"每轨各自算等级，取较高者"。
"""

import pytest

from src.baseline.scaler_utils import TRACK_SLEEP, TRACK_SOCIAL
from src.risk.judge import build_mpdd_evidence, judge_risk_level


def day(
    day_key: str,
    sleep_score: float = 0.5,
    sleep_dev: bool = False,
    social_score: float = 0.5,
    social_dev: bool = False,
    quality: str = "valid",
    sleep_threshold: float = 1.0,
    social_threshold: float = 1.0,
) -> dict:
    """构造一天的双轨推理结果"""
    return {
        "elder_id": "E001",
        "day_key": day_key,
        "data_quality": quality,
        "is_deviation": sleep_dev or social_dev,
        TRACK_SLEEP: {
            "track": TRACK_SLEEP,
            "status": "success",
            "signed_available": True,
            "anomaly_score": sleep_score,
            # ★ 必须带动态阈值：_severity 在 threshold=0 时退回绝对分（score），
            #   幅度门槛测试就永远跑不到真实的 severity = score/threshold 除法——
            #   这正是 test_regression_2026_07_31.py:33-35 记过的陷阱
            #   ("手搓字典时漏字段正是上一轮'绿得没有意义'的成因")。
            "dynamic_threshold": sleep_threshold,
            "static_threshold": sleep_threshold,
            "ewma_threshold": sleep_threshold,
            "is_deviation": sleep_dev,
            "signed_z": {},
        },
        TRACK_SOCIAL: {
            "track": TRACK_SOCIAL,
            "status": "success",
            "signed_available": True,
            "anomaly_score": social_score,
            "dynamic_threshold": social_threshold,
            "static_threshold": social_threshold,
            "ewma_threshold": social_threshold,
            "is_deviation": social_dev,
            "signed_z": {},
        },
    }


class TestJudgeRiskLevel:
    def test_no_data(self):
        result = judge_risk_level("E001", daily_results=[])
        assert result["risk_level"] == 0
        assert result["risk_label"] == "正常"

    def test_normal_single_day(self):
        result = judge_risk_level("E001", daily_results=[day("2026-08-01")])
        assert result["risk_level"] == 0

    def test_single_deviation_attention(self):
        results = [day("2026-08-01", sleep_score=1.2, sleep_dev=True)]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 1
        assert result["risk_label"] == "关注"

    def test_three_consecutive_warning(self):
        results = [
            day(f"2026-08-0{i}", sleep_score=1.2 + i * 0.1, sleep_dev=True)
            for i in range(1, 4)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 2
        assert result["risk_label"] == "提醒"

    def test_five_consecutive_severe(self):
        results = [
            day(f"2026-08-0{i}", sleep_score=1.2 + i * 0.1, sleep_dev=True)
            for i in range(1, 6)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 3
        assert result["risk_label"] == "严重"

    def test_intermittent_deviation(self):
        results = [
            day("2026-08-01", sleep_score=1.2, sleep_dev=True),
            day("2026-08-02", sleep_score=0.5, sleep_dev=False),
            day("2026-08-03", sleep_score=1.3, sleep_dev=True),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] <= 1

    def test_high_single_spike_attention(self):
        results = [day("2026-08-01", sleep_score=5.0, sleep_dev=True)]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] >= 1

    def test_recovery_after_deviation(self):
        results = [
            day("2026-08-01", sleep_score=1.2, sleep_dev=True),
            day("2026-08-02", sleep_score=1.3, sleep_dev=True),
            day("2026-08-03", sleep_score=0.4),
            day("2026-08-04", sleep_score=0.3),
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 0

    def test_low_amplitude_streak_not_severe(self):
        """连续5天但均为低幅度擦线偏离 → 不应升到严重级（幅度门槛）

        回归测试：Level 3 会触发社区网格员介入，不能仅凭连续天数触发。
        """
        results = [
            day(f"2026-08-0{i}", sleep_score=0.6, sleep_dev=True) for i in range(1, 6)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] < 3

    def test_high_amplitude_streak_severe(self):
        results = [
            day(f"2026-08-0{i}", sleep_score=1.5, sleep_dev=True) for i in range(1, 6)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 3


class TestPerTrackJudgment:
    """★ 每轨独立判级，取较高者"""

    def test_per_track_levels_reported(self):
        results = [
            day(f"2026-08-0{i}", sleep_score=1.5, sleep_dev=True, social_score=0.4)
            for i in range(1, 4)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["per_track"][TRACK_SLEEP]["risk_level"] == 2
        assert result["per_track"][TRACK_SOCIAL]["risk_level"] == 0
        assert result["risk_level"] == 2, "整体取较高者"

    def test_social_only_deviation_still_escalates(self):
        """睡眠正常、社交连续偏离 → 整体等级应由社交轨决定"""
        results = [
            day(f"2026-08-0{i}", sleep_score=0.3, social_score=1.6, social_dev=True)
            for i in range(1, 4)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["per_track"][TRACK_SLEEP]["risk_level"] == 0
        assert result["per_track"][TRACK_SOCIAL]["risk_level"] == 2
        assert result["risk_level"] == 2

    def test_scores_not_averaged_across_tracks(self):
        """两轨分数不得相加或平均——尺度不同，混算无统计意义。

        睡眠轨高分 + 社交轨零分，若被平均会掉到门槛以下、漏掉睡眠预警。
        """
        results = [
            day(f"2026-08-0{i}", sleep_score=2.0, sleep_dev=True, social_score=0.0)
            for i in range(1, 6)
        ]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 3, "睡眠轨自身够严重，不应被社交轨的低分拉平"
        assert result["per_track"][TRACK_SLEEP]["avg_anomaly"] == 2.0

    def test_unavailable_track_not_counted_as_normal(self):
        """某轨 cold_start → 标记不可评估，而不是当成 0 级拉低整体"""
        results = []
        for i in range(1, 4):
            d = day(f"2026-08-0{i}", social_score=1.6, social_dev=True)
            d[TRACK_SLEEP] = {"track": TRACK_SLEEP, "status": "cold_start",
                              "signed_available": False}
            results.append(d)
        result = judge_risk_level("E001", daily_results=results)
        assert not result["per_track"][TRACK_SLEEP]["evaluable"]
        assert result["per_track"][TRACK_SOCIAL]["evaluable"]
        assert result["risk_level"] == 2

    def test_both_tracks_unavailable(self):
        results = [{
            "elder_id": "E001", "day_key": "2026-08-01",
            TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start"},
            TRACK_SOCIAL: {"track": TRACK_SOCIAL, "status": "cold_start"},
        }]
        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 0


class TestRecommendationWording:
    """措辞约束：只说行为观察，不下诊断"""

    def test_no_diagnostic_terms(self):
        results = [
            day(f"2026-08-0{i}", sleep_score=2.0, sleep_dev=True) for i in range(1, 6)
        ]
        rec = judge_risk_level("E001", daily_results=results)["recommendation"]
        for banned in ("抑郁症", "睡眠障碍", "孤独症", "确诊"):
            assert banned not in rec

    def test_normal_day_wording(self):
        rec = judge_risk_level("E001", daily_results=[day("2026-08-01")])["recommendation"]
        assert "稳定" in rec


class TestMPDDEvidence:
    def test_contract_keys_and_note(self):
        daily = day("2026-08-05", sleep_score=1.8, sleep_dev=True)
        daily[TRACK_SLEEP]["signed_z"] = {"sleep_efficiency": -2.1, "waso_min": 2.7}
        daily[TRACK_SOCIAL]["signed_z"] = {
            "rar_amplitude": -2.4, "rar_iv": 2.1, "copresence_min": -0.5,
        }
        risk = judge_risk_level("E001", daily_results=[daily])

        payload = build_mpdd_evidence("E001", "2026-08-05", daily, risk)
        assert payload["elder_id"] == "E001"
        assert payload["day_key"] == "2026-08-05"
        assert payload["sleep_evidence"]["signed_z"]["sleep_efficiency"] == -2.1
        # 单向契约必须写明
        assert "不消费" in payload["note"]

    def test_circadian_block_excludes_contact_features(self):
        daily = day("2026-08-05")
        daily[TRACK_SOCIAL]["signed_z"] = {
            "rar_amplitude": -2.4, "rar_iv": 2.1,
            "copresence_min": -1.0, "out_of_home_min": -0.8, "activity_counts": -0.9,
        }
        risk = judge_risk_level("E001", daily_results=[daily])
        payload = build_mpdd_evidence("E001", "2026-08-05", daily, risk)

        assert set(payload["circadian_evidence"]["signed_z"]) == {"rar_amplitude", "rar_iv"}
        assert set(payload["social_evidence"]["signed_z"]) == {
            "copresence_min", "out_of_home_min", "activity_counts"
        }

    def test_quality_downgrade_on_cold_start(self):
        daily = day("2026-08-05")
        daily[TRACK_SLEEP] = {"track": TRACK_SLEEP, "status": "cold_start"}
        risk = judge_risk_level("E001", daily_results=[daily])
        payload = build_mpdd_evidence("E001", "2026-08-05", daily, risk)
        assert payload["sleep_evidence"]["quality"] == "cold_start"


class TestDirectionGate:
    """
    方向闸门：anomaly_score 用 abs 残差、方向无关，
    "睡得明显更好" 与 "明显更差" 得同样高的分。
    L2/L3 会触发家属提醒，不能因为老人好转就发"严重风险"。
    """

    def _day(self, day_key: str, z: dict, score: float = 1.8) -> dict:
        d = day(day_key, sleep_score=score, sleep_dev=True)
        d[TRACK_SLEEP]["signed_z"] = z
        return d

    def test_all_improvement_capped_at_attention(self):
        # sleep_efficiency 方向为 down（跌才是坏）；这里大幅上涨 = 好转
        good = {"sleep_efficiency": 3.5, "waso_min": -3.0}
        results = [self._day(f"2026-08-0{i}", good) for i in range(1, 7)]
        result = judge_risk_level("E001", daily_results=results)

        assert result["risk_level"] == 1, "全是好转不该升到提醒/严重"
        assert result["per_track"][TRACK_SLEEP]["adverse_direction"] is False

    def test_adverse_movement_still_escalates(self):
        bad = {"sleep_efficiency": -3.5, "waso_min": 3.0}
        results = [self._day(f"2026-08-0{i}", bad) for i in range(1, 7)]
        result = judge_risk_level("E001", daily_results=results)

        assert result["risk_level"] == 3
        assert result["per_track"][TRACK_SLEEP]["adverse_direction"] is True

    def test_mixed_direction_escalates(self):
        """有一维朝坏方向就够了，不要求全部朝坏"""
        mixed = {"sleep_efficiency": 3.5, "waso_min": 3.0}
        results = [self._day(f"2026-08-0{i}", mixed) for i in range(1, 7)]
        assert judge_risk_level("E001", daily_results=results)["risk_level"] == 3

    def test_missing_signed_z_does_not_cap(self):
        """判不出方向时保留原等级——数据缺失不是好转的证据"""
        results = [self._day(f"2026-08-0{i}", {}) for i in range(1, 7)]
        assert judge_risk_level("E001", daily_results=results)["risk_level"] == 3

    def test_signed_unavailable_does_not_cap(self):
        results = []
        for i in range(1, 7):
            d = self._day(f"2026-08-0{i}", {"sleep_efficiency": 3.5})
            d[TRACK_SLEEP]["signed_available"] = False
            results.append(d)
        assert judge_risk_level("E001", daily_results=results)["risk_level"] == 3

    def test_no_feature_crosses_threshold_caps(self):
        """
        总分够高但没有任何一维越过 ±1σ → 拿不到"朝坏方向"的正面证据，封顶 L1。

        这种情形是偏离被摊薄在各维上（每维都只擦线）。此时发"严重风险"
        缺乏可解释的依据，压到"关注"更诚实。
        """
        weak = {"sleep_efficiency": 0.4}
        results = [self._day(f"2026-08-0{i}", weak) for i in range(1, 7)]
        assert judge_risk_level("E001", daily_results=results)["risk_level"] == 1

    def test_single_adverse_day_in_improving_streak_still_caps(self):
        """
        一周全面好转中夹一天"任意方向"维的抖动，不该放行 L3。

        实测触发点：sleep_onset_clock（direction=any）单日 2.26σ，
        其余 6 天全是好转。升级本身是"持续偏离"换来的，
        方向证据也该持续，不能靠一天。
        """
        good = {"sleep_efficiency": 3.0, "waso_min": -3.0}
        results = [self._day(f"2026-08-0{i}", good) for i in range(1, 8)]
        # 第 2 天插一个越过 any 门槛的就寝时间抖动
        results[1][TRACK_SLEEP]["signed_z"] = {**good, "sleep_onset_clock": 2.3}

        result = judge_risk_level("E001", daily_results=results)
        assert result["risk_level"] == 1
        assert result["per_track"][TRACK_SLEEP]["adverse_direction"] is False

    def test_majority_adverse_days_escalates(self):
        """过半天数朝坏方向 → 正常升级"""
        bad = {"sleep_efficiency": -3.0}
        good = {"sleep_efficiency": 3.0}
        results = [self._day(f"2026-08-0{i}", bad) for i in range(1, 6)]
        results += [self._day(f"2026-08-0{i}", good) for i in range(6, 8)]
        assert judge_risk_level("E001", daily_results=results)["risk_level"] == 3

    def test_any_direction_needs_higher_bar(self):
        """
        direction=any 的维不含好/坏信息，门槛更高（2σ）。
        1.3σ 的就寝时间抖动是正常生活波动，不算"变坏的证据"。
        """
        good = {"sleep_efficiency": 3.0}
        marginal = [self._day(f"2026-08-0{i}", {**good, "sleep_onset_clock": 1.3})
                    for i in range(1, 8)]
        assert judge_risk_level("E001", daily_results=marginal)["risk_level"] == 1

        clear = [self._day(f"2026-08-0{i}", {**good, "sleep_onset_clock": 2.5})
                 for i in range(1, 8)]
        assert judge_risk_level("E001", daily_results=clear)["risk_level"] == 3

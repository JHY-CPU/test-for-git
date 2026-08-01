"""
端到端集成测试：双轨全流程

在临时目录里跑真实管道，而不是只验证组件接口：
    生成 60 天双轨数据 → 双轨建档 → 逐日推理 → 风险判定 → 微调

核心验证点是**双轨信号隔离**：睡眠异常段社交轨应保持正常，反之亦然。
若两轨同时报警，说明残差串轨了。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import (
    SLEEP_FEATURE_DIM,
    SOCIAL_FEATURE_DIM,
    TRACK_SLEEP,
    TRACK_SOCIAL,
)

ELDER = "E001"
START = "2026-07-01"
N_DAYS = 60


# sandbox fixture 已上移到 tests/conftest.py（多个测试文件都要用真实 data/ 布局，
# 各写一份会漂）。这里直接按名字取用。


@pytest.fixture
def simulated_data(sandbox):
    """用生产用的生成器造数据，保证测试与实际演示走同一条路径"""
    from scripts.generate_simulation_data import generate_all_data

    generate_all_data(sandbox, n_days=N_DAYS, start_date=START)
    return sandbox


def day_key_of(day: int) -> str:
    return (datetime.strptime(START, "%Y-%m-%d") + timedelta(days=day - 1)).strftime("%Y-%m-%d")


class TestFeatureContract:
    def test_track_dims(self):
        assert SLEEP_FEATURE_DIM == 8
        assert SOCIAL_FEATURE_DIM == 5

    def test_weights_cover_all_features(self):
        """权重配置必须覆盖两轨全部特征，缺任何一个都应报错而不是默认 1.0"""
        from src.utils.io import get_feature_directions, get_feature_weight_array

        for track, dim in ((TRACK_SLEEP, 8), (TRACK_SOCIAL, 5)):
            w = get_feature_weight_array(track)
            d = get_feature_directions(track)
            assert w.shape == (dim,)
            assert len(d) == dim
            assert np.all(w > 0)

    def test_no_legacy_features_remain(self):
        """已删除的旧特征不得残留在任何轨里"""
        from src.baseline.scaler_utils import SLEEP_FEATURES, SOCIAL_FEATURES

        removed = {
            "sfi", "hrv_rmssd", "daily_activity", "social_turns",
            "sad_ratio", "avg_speed", "pitch_variability", "distress_events",
            "voice_active_min", "first_activity_clock", "day_sin", "day_cos",
        }
        assert not (set(SLEEP_FEATURES) & removed)
        assert not (set(SOCIAL_FEATURES) & removed)


class TestDataGeneration:
    def test_both_csvs_written(self, simulated_data):
        base = simulated_data / "data" / "features" / ELDER
        assert (base / "features_sleep.csv").exists()
        assert (base / "features_social.csv").exists()

    def test_no_legacy_csv(self, simulated_data):
        """旧的单轨 features.csv 不应再被生成"""
        assert not (simulated_data / "data" / "features" / ELDER / "features.csv").exists()

    def test_anomaly_injected_in_right_track(self, simulated_data):
        from src.utils.io import load_features_csv

        sleep = load_features_csv(ELDER, TRACK_SLEEP)
        social = load_features_csv(ELDER, TRACK_SOCIAL)

        # 睡眠异常段（40-46）SE 应显著低于建档期
        base_se = sleep["sleep_efficiency"].iloc[:35].mean()
        anom_se = sleep["sleep_efficiency"].iloc[39:46].mean()
        assert anom_se < base_se - 0.1, f"睡眠异常未注入: {base_se:.3f} → {anom_se:.3f}"

        # 社交异常段（50-58）三个必需维都应下降
        for feat in ("copresence_min", "out_of_home_min", "activity_counts"):
            base = social[feat].iloc[:35].mean()
            anom = social[feat].iloc[49:58].mean()
            assert anom < base * 0.6, f"{feat} 未下降: {base:.1f} → {anom:.1f}"

        # 节律塌陷：RA↓ 且 IV↑
        assert social["rar_amplitude"].iloc[49:58].mean() < social["rar_amplitude"].iloc[:35].mean()
        assert social["rar_iv"].iloc[49:58].mean() > social["rar_iv"].iloc[:35].mean()

    def test_weekend_effect_present(self, simulated_data):
        """★ 周末效应必须存在，否则测不出分池 EWMA 的收益"""
        from src.utils.io import load_features_csv

        social = load_features_csv(ELDER, TRACK_SOCIAL).iloc[:35]
        dates = pd_to_datetime(social["day_key"])
        weekend = dates.dt.weekday >= 5
        assert social.loc[weekend, "copresence_min"].mean() > \
               social.loc[~weekend, "copresence_min"].mean() * 1.5


def pd_to_datetime(series):
    import pandas as pd
    return pd.to_datetime(series)


class TestTrainingAndInference:
    def test_both_tracks_train(self, simulated_data):
        from src.baseline.trainer import train_all_tracks

        results = train_all_tracks(ELDER)
        assert results == {TRACK_SLEEP: "success", TRACK_SOCIAL: "success"}

        baseline_dir = simulated_data / "data" / "baselines" / ELDER
        for name in (
            "gru_sleep.pth", "gru_social.pth",
            "scaler_sleep.pkl", "scaler_social.pkl",
            "residual_stats_sleep.pkl", "residual_stats_social.pkl",
            "ewma_sleep.pkl", "ewma_social_weekday.pkl", "ewma_social_weekend.pkl",
            "baseline_meta.json",
        ):
            assert (baseline_dir / name).exists(), f"缺少基线文件: {name}"

    def test_residual_stats_have_both_kinds(self, simulated_data):
        """★ 双残差契约：signed 与 abs 两套统计都必须落盘"""
        from src.baseline.trainer import train_initial_baseline
        from src.utils.io import load_residual_stats

        train_initial_baseline(ELDER, TRACK_SLEEP)
        stats = load_residual_stats(ELDER, TRACK_SLEEP)
        assert set(stats) == {"signed", "abs"}
        assert stats["signed"]["std"].shape == (8,)
        assert np.all(stats["abs"]["mean"] >= 0), "abs 均值恒非负"

    def test_threshold_from_holdout_not_train(self, simulated_data):
        """★ 阈值必须用留出段估。

        留出段残差应大于训练段残差——若相反或相等，说明阈值仍在用训练集，
        那条"残差趋零→阈值分母趋零→疯狂误报"的链条就没被切断。
        """
        import torch
        from src.baseline.gru_model import PersonalBaselineGRU
        from src.baseline.trainer import _build_windows, train_initial_baseline
        from src.baseline.scaler_utils import get_feature_names, load_scaler
        from src.utils.io import get_scaler_path, load_features_csv, load_residual_stats

        model, scaler, stats, _ = train_initial_baseline(ELDER, TRACK_SLEEP)

        df = load_features_csv(ELDER, TRACK_SLEEP)
        df = df[df["data_quality"] == "valid"].sort_values("day_key").iloc[:35]
        data = df[get_feature_names(TRACK_SLEEP)].to_numpy(dtype=np.float64)
        norm = scaler.transform(data)

        X_tr, y_tr, _ = _build_windows(norm, 7, 7, 28)
        model.eval()
        with torch.no_grad():
            train_abs = np.abs((y_tr - model(X_tr)).numpy()).mean()

        holdout_abs = stats["abs"]["mean"].mean()
        assert holdout_abs > train_abs, (
            f"留出段残差({holdout_abs:.4f})应大于训练段({train_abs:.4f})，"
            f"否则阈值仍取自训练集"
        )

    def test_ewma_social_pools_both_populated(self, simulated_data):
        from src.baseline.ewma import TrackEWMAPools
        from src.baseline.trainer import train_initial_baseline
        from src.utils.io import get_baseline_dir

        train_initial_baseline(ELDER, TRACK_SOCIAL)
        pools = TrackEWMAPools.load(get_baseline_dir(ELDER), TRACK_SOCIAL)
        assert pools.n_samples(is_weekend=False) > 0
        assert pools.n_samples(is_weekend=True) > 0


class TestTrackIsolation:
    """★ 本次重构的核心收益：一轨异常不污染另一轨"""

    @pytest.fixture
    def run_timeline(self, simulated_data):
        from src.baseline.trainer import train_all_tracks
        from src.scheduler.daily_job import run_daily_pipeline

        train_all_tracks(ELDER)

        timeline = {}
        for day in range(36, N_DAYS + 1):
            dk = day_key_of(day)
            result = run_daily_pipeline(ELDER, dk)
            inf = result.get("inference_result") or {}
            timeline[day] = {
                "sleep": (inf.get(TRACK_SLEEP) or {}).get("is_deviation", False),
                "social": (inf.get(TRACK_SOCIAL) or {}).get("is_deviation", False),
                "risk": (result.get("risk_result") or {}).get("risk_level", 0),
                "types": [
                    r["risk_key"]
                    for r in (result.get("risk_result") or {}).get("risk_types", [])
                ],
            }
        return timeline

    def test_sleep_anomaly_detected(self, run_timeline):
        flagged = [d for d in range(40, 47) if run_timeline[d]["sleep"]]
        assert len(flagged) >= 6, f"睡眠异常段应几乎全部检出，实际 {flagged}"

    def test_social_quiet_during_sleep_anomaly(self, run_timeline):
        """睡眠恶化期社交轨不应大面积报警"""
        noisy = [d for d in range(41, 47) if run_timeline[d]["social"]]
        assert len(noisy) <= 1, f"睡眠异常期社交轨串轨: {noisy}"

    def test_social_anomaly_detected(self, run_timeline):
        flagged = [d for d in range(50, 59) if run_timeline[d]["social"]]
        assert len(flagged) >= 8, f"社交异常段应几乎全部检出，实际 {flagged}"

    def test_sleep_quiet_during_social_anomaly(self, run_timeline):
        """社交退缩期睡眠轨不应报警"""
        noisy = [d for d in range(50, 59) if run_timeline[d]["sleep"]]
        assert noisy == [], f"社交异常期睡眠轨串轨: {noisy}"

    def test_correct_risk_type_activated(self, run_timeline):
        """睡眠段激活睡眠类型，社交段激活社会类型，不能互相冒名"""
        sleep_types = {t for d in range(42, 47) for t in run_timeline[d]["types"]}
        social_types = {t for d in range(54, 59) for t in run_timeline[d]["types"]}
        assert "sleep_stability" in sleep_types
        assert "social_decline" not in sleep_types
        assert "social_decline" in social_types
        assert "sleep_stability" not in social_types

    def test_recovery_returns_to_normal(self, run_timeline):
        """恢复期两轨都应回到不偏离"""
        for day in (59, 60):
            assert not run_timeline[day]["sleep"], f"Day{day} 睡眠轨未恢复"
            assert not run_timeline[day]["social"], f"Day{day} 社交轨未恢复"

    def test_risk_escalates_then_clears(self, run_timeline):
        """等级应在异常段升上去、恢复期落回来"""
        assert max(run_timeline[d]["risk"] for d in range(42, 47)) >= 2
        assert max(run_timeline[d]["risk"] for d in range(54, 59)) >= 2
        assert run_timeline[60]["risk"] <= 1


class TestDegradedOperation:
    def test_sleep_offline_social_still_runs(self, simulated_data):
        """★ 故障隔离：小贝壳没有数据时社交轨照常出结果"""
        from src.baseline.trainer import train_all_tracks
        from src.scheduler.daily_job import run_daily_pipeline

        train_all_tracks(ELDER)
        dk = day_key_of(40)

        result = run_daily_pipeline(
            ELDER, dk,
            raw_data={"sleep": None, "activity": _activity_payload(), "camera": {"copresence_min": 50.0}},
        )
        assert result["track_quality"][TRACK_SLEEP] == "insufficient"
        assert result["track_quality"][TRACK_SOCIAL] == "valid"
        social = (result.get("inference_result") or {}).get(TRACK_SOCIAL)
        assert social is not None
        assert social["status"] in ("success", "observation")

    def test_camera_offline_degrades_social_only(self, simulated_data):
        """★ copresence 缺失（禁止填充）→ 社交轨降级，睡眠轨不受影响。

        降级而非作废：剩下 4 维对活动/节律仍有效。但绝不能算 valid——
        那会让系统拿活动量继续输出"社会连接正常"，而它根本测不到社会接触。
        """
        from src.baseline.trainer import train_all_tracks
        from src.data_pipeline.validator import counts_toward_consecutive
        from src.scheduler.daily_job import run_daily_pipeline

        train_all_tracks(ELDER)
        dk = day_key_of(41)

        result = run_daily_pipeline(
            ELDER, dk,
            raw_data={"sleep": _sleep_payload(), "activity": _activity_payload(), "camera": None},
        )
        assert result["track_quality"][TRACK_SLEEP] == "valid"
        assert result["track_quality"][TRACK_SOCIAL] == "degraded"
        # 降级日不得计入连续偏离天数，避免不完整证据攒出预警
        assert not counts_toward_consecutive(result["track_quality"][TRACK_SOCIAL])

    def test_degraded_quality_reaches_inference_log_per_track(self, simulated_data):
        """★ 回归：degraded 标记必须**随推理结果落盘**，而不是只进 features CSV。

        历史缺陷：validator 判得对，但标记停在 features_social.csv，没进
        data/logs/daily_inference/*.json。rules._counts_toward_consecutive 读不到
        字段就走"保守当作有效"分支恒返回 True —— "degraded 日既不累加也不打断"
        这条设计在生产链路里从未生效，传感器抖动照样能攒成预警。

        本用例断言的是 **rules 层真的跳过了这天**，而不是上一个用例那样只断言
        validator 的辅助函数——断言打错层正是这个缺陷能长期存活的原因。
        """
        import json

        from src.baseline.trainer import train_all_tracks
        from src.risk.rules import _counts_toward_consecutive
        from src.scheduler.daily_job import run_daily_pipeline
        from src.utils.io import get_log_dir

        train_all_tracks(ELDER)
        dk = day_key_of(42)

        run_daily_pipeline(
            ELDER, dk,
            raw_data={"sleep": _sleep_payload(), "activity": _activity_payload(), "camera": None},
        )

        log = get_log_dir("daily_inference") / f"{ELDER}_{dk}.json"
        assert log.exists()
        payload = json.loads(log.read_text(encoding="utf-8"))

        assert payload[TRACK_SOCIAL]["data_quality"] == "degraded"
        assert payload[TRACK_SLEEP]["data_quality"] == "valid"

        # 按轨判定：社交轨的规则跳过这天，睡眠轨的规则不受牵连——
        # 否则双轨的故障隔离会在判定层被粘回去。
        assert not _counts_toward_consecutive(payload, frozenset({TRACK_SOCIAL}))
        assert _counts_toward_consecutive(payload, frozenset({TRACK_SLEEP}))

    def test_social_contact_rule_cannot_fire_without_copresence(self, simulated_data):
        """copresence 缺失时「社会连接减弱」必须无法触发，而不是用剩下两维凑合判定"""
        from src.risk.rules import classify_risk_type

        track_results = {
            TRACK_SLEEP: {"track": TRACK_SLEEP, "status": "cold_start",
                          "signed_available": False},
            TRACK_SOCIAL: {
                "track": TRACK_SOCIAL, "status": "success", "signed_available": True,
                # copresence 不在 signed_z 里（缺失）
                "signed_z": {"out_of_home_min": -4.0, "activity_counts": -4.0,
                             "rar_amplitude": -0.1, "rar_iv": 0.1},
            },
        }
        results = classify_risk_type(track_results, daily_results=[])
        social = next(r for r in results if r["risk_key"] == "social_decline")
        assert not social["qualifies"], "缺 copresence 时不得凭其余两维判定社会连接减弱"

    def test_cold_start_fallback_used_before_baseline(self, simulated_data):
        """建档前应走稳健兜底而不是完全不检测"""
        from src.scheduler.daily_job import run_daily_pipeline

        # 未训练任何基线
        result = run_daily_pipeline(ELDER, day_key_of(20))
        inf = result.get("inference_result") or {}
        statuses = {t: (inf.get(t) or {}).get("status") for t in (TRACK_SLEEP, TRACK_SOCIAL)}
        assert all(s == "cold_start_fallback" for s in statuses.values()), statuses

    def test_cold_start_fallback_actually_alerts(self, simulated_data):
        """★ 回归：建档期出现严重异常时必须真的出等级，而不是只写日志。

        历史缺陷：兜底算出 is_deviation=True、给了带方向的 signed_z，但
        judge._judge_single_track 与 rules._collect_signed_z 的状态白名单只收
        ("success","observation")，把 cold_start_fallback 整个丢掉。实测连续 6 天
        anomaly_score 166→58（阈值 3.0）、偏离 6/6，risk_level 全是 0——
        整个 35 天建档期一条预警都发不出，而 cold_start_fallback.py 的存在
        理由正是"消除建档期监测盲区"。盲区在检测层补上了，预警层原封不动。

        上一个用例只断言 status 流转，从没断言过能出等级——这就是缺口能存活的原因。
        """
        from src.scheduler.daily_job import run_daily_pipeline

        bad_sleep = {
            "sleep_efficiency": 0.52, "waso_min": 140.0, "sol_min": 85.0,
            "bed_exit_count": 6.0, "deep_sleep_ratio": 0.05,
            "sleep_onset_clock": 300.0, "night_hr_mean": 78.0, "daytime_nap_min": 150.0,
        }

        levels = []
        for d in range(15, 20):          # 建档期内（< build_days=35），连续 5 天
            result = run_daily_pipeline(
                ELDER, day_key_of(d),
                raw_data={"sleep": bad_sleep, "activity": _activity_payload(),
                          "camera": {"copresence_min": 75.0}},
            )
            sleep = (result.get("inference_result") or {}).get(TRACK_SLEEP) or {}
            assert sleep.get("status") == "cold_start_fallback", sleep.get("status")
            assert sleep.get("is_deviation"), "严重异常在兜底层就该判偏离"
            levels.append((result.get("risk_result") or {}).get("risk_level", 0))

        assert levels[0] >= 1, f"建档期首个严重异常日应至少出关注级，实际 {levels}"
        assert max(levels) >= 2, f"连续 5 天严重异常应升到提醒级以上，实际 {levels}"

    def test_cold_start_fallback_normal_days_stay_quiet(self, simulated_data):
        """★ 兜底轨的正常天不得误报——两套尺度混用会在这里翻车。

        兜底的 anomaly_score 是稳健加权 |z|（阈值 = fallback_sigma = 3.0），
        正常天实测能到 1.77；而幅度门槛若沿用照 GRU 残差尺度定的绝对常数
        （high_spike=1.5），建档期会天天误报 L1。这正是幅度门槛必须改成
        severity（分数/自身阈值）的原因。
        """
        from src.scheduler.daily_job import run_daily_pipeline

        for d in range(15, 19):
            result = run_daily_pipeline(ELDER, day_key_of(d))
            sleep = (result.get("inference_result") or {}).get(TRACK_SLEEP) or {}
            assert sleep.get("status") == "cold_start_fallback"
            assert not sleep.get("is_deviation"), f"day{d} 正常天不该判偏离"
            level = (result.get("risk_result") or {}).get("risk_level", 0)
            assert level == 0, f"day{d} 兜底期正常天不该出等级，实际 L{level}"


def _sleep_payload() -> dict:
    return {
        "sleep_efficiency": 0.88, "waso_min": 34.0, "sol_min": 17.0,
        "bed_exit_count": 2.0, "deep_sleep_ratio": 0.22,
        "sleep_onset_clock": 148.0, "night_hr_mean": 62.0, "daytime_nap_min": 38.0,
    }


def _activity_payload() -> dict:
    return {
        "activity_counts": 185.0, "out_of_home_min": 92.0,
        "rar_amplitude": 0.92, "rar_iv": 0.29,
    }


class TestMPDDContract:
    def test_contract_shape(self, simulated_data):
        from src.baseline.trainer import train_all_tracks
        from src.risk.judge import build_mpdd_evidence, quick_judge
        from src.scheduler.daily_job import run_daily_pipeline

        train_all_tracks(ELDER)
        dk = day_key_of(44)
        result = run_daily_pipeline(ELDER, dk)

        payload = build_mpdd_evidence(
            ELDER, dk, result["inference_result"], result["risk_result"] or {}
        )
        assert payload["schema_version"] == "2.1.0"
        assert set(payload) >= {
            "sleep_evidence", "circadian_evidence", "social_evidence", "day_key",
        }
        # 节律块只带 RA/IV，不混入社交接触维
        assert set(payload["circadian_evidence"]["signed_z"]) <= {"rar_amplitude", "rar_iv"}
        assert "copresence_min" not in payload["circadian_evidence"]["signed_z"]
        assert "copresence_min" in payload["social_evidence"]["signed_z"]

    def test_signed_z_sign_convention(self, simulated_data):
        """★ signed_z = observed − predicted：社交退缩期 copresence 应为负"""
        from src.baseline.trainer import train_all_tracks
        from src.scheduler.daily_job import run_daily_pipeline

        train_all_tracks(ELDER)
        for day in range(50, 56):
            run_daily_pipeline(ELDER, day_key_of(day))
        result = run_daily_pipeline(ELDER, day_key_of(56))

        social = (result["inference_result"] or {}).get(TRACK_SOCIAL) or {}
        assert social["signed_z"]["copresence_min"] < 0, "共处时长下降应为负 z"


class TestReportAndAlert:
    def test_alert_levels(self):
        from src.risk.alert import trigger_alert

        assert not trigger_alert(ELDER, 1, [{"risk_type": "睡眠稳定性偏离"}])["alerted"]
        assert trigger_alert(ELDER, 2, [{"risk_type": "睡眠稳定性偏离"}])["alerted"]

    def test_rule_based_report(self):
        from src.report.templates import generate_rule_based_report

        report = generate_rule_based_report(
            elder_id=ELDER,
            week_start="2026-08-01",
            week_end="2026-08-07",
            social_trend="下降",
            sleep_trend="下降",
            activity_trend="平稳",
            deviation_days=3,
            risk_label="提醒",
            risk_types=["睡眠稳定性偏离"],
        )
        assert ELDER in report
        # 措辞约束：不得出现诊断名
        for banned in ("抑郁症", "睡眠障碍", "孤独症"):
            assert banned not in report

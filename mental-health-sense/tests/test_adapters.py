"""
传感器适配器单元测试

重点覆盖两件容易写错的事：
    1. day_key 时间窗派生（修掉 night_of 循环定义）
    2. copresence 去抖与 ROI 过滤（社会接触唯一来源，不准就整轨不准）
"""

import numpy as np
import pytest

from src.baseline.scaler_utils import SLEEP_FEATURES
from src.data_pipeline.adapters.camera import (
    CameraAdapter,
    check_cloud_calibration,
    compute_copresence_minutes,
    filter_tv_roi_detections,
    smooth_person_counts,
)
from src.data_pipeline.adapters.circadian import (
    build_hourly_counts,
    compute_interdaily_stability,
    compute_m10_l5,
    compute_rar_amplitude,
    compute_rar_iv,
)
from src.data_pipeline.adapters.ezviz_events import (
    ALARM_TYPES_ACTIVITY,
    EzvizEventAdapter,
    compute_out_of_home,
    derive_day_window,
    derive_social_activity_features,
    filter_activity_events,
)
from src.data_pipeline.adapters.xiaobeike import (
    XiaobeikeAdapter,
    derive_sleep_features,
    normalize_whst_report,
    onset_clock_minutes,
    search_window,
)


class TestSearchWindow:
    """★ day_key 索引：纯日历，与任何测量值无关"""

    def test_window_is_fixed_calendar_range(self):
        start, end = search_window("2026-08-02")
        assert (start.month, start.day, start.hour) == (8, 1, 18)
        assert (end.month, end.day, end.hour) == (8, 2, 14)

    def test_window_independent_of_measurements(self):
        """同一 day_key 的窗口恒定——这是修掉循环定义的关键性质"""
        assert search_window("2026-08-02") == search_window("2026-08-02")

    def test_month_boundary(self):
        start, end = search_window("2026-09-01")
        assert (start.month, start.day) == (8, 31)
        assert (end.month, end.day) == (9, 1)


class TestOnsetClock:
    """入睡时刻编码必须跨午夜连续，否则相位漂移会被读成跳变"""

    def test_evening_onset_positive(self):
        from datetime import datetime
        assert onset_clock_minutes(datetime(2026, 8, 1, 22, 30)) == 150.0

    def test_before_reference_negative(self):
        from datetime import datetime
        assert onset_clock_minutes(datetime(2026, 8, 1, 19, 30)) == -30.0

    def test_after_midnight_continues_upward(self):
        """★ 00:30 入睡应得 270（继续增大），不是 −1170 之类的跳变"""
        from datetime import datetime
        assert onset_clock_minutes(datetime(2026, 8, 2, 0, 30)) == 270.0

    def test_monotonic_across_midnight(self):
        from datetime import datetime, timedelta
        base = datetime(2026, 8, 1, 22, 0)
        values = [onset_clock_minutes(base + timedelta(minutes=30 * i)) for i in range(8)]
        assert values == sorted(values), f"跨午夜应单调递增: {values}"


def _normal_night_timeline() -> list[dict]:
    return [
        {"stage": "WAKEFUL", "ts": "2026-08-01T22:30:00"},
        {"stage": "LIGHT", "ts": "2026-08-01T22:50:00"},
        {"stage": "DEEP", "ts": "2026-08-02T00:20:00"},
        {"stage": "WAKEFUL", "ts": "2026-08-02T01:30:00"},
        {"stage": "UNKNOWN", "ts": "2026-08-02T01:45:00"},
        {"stage": "LIGHT", "ts": "2026-08-02T01:55:00"},
        {"stage": "DEEP", "ts": "2026-08-02T04:00:00"},
        {"stage": "UNKNOWN", "ts": "2026-08-02T06:30:00"},
    ]


class TestDeriveSleepFeatures:
    def test_all_eight_features_keys(self):
        f = derive_sleep_features("2026-08-02", _normal_night_timeline())
        assert set(f) == set(SLEEP_FEATURES)

    def test_normal_night_values_sane(self):
        f = derive_sleep_features("2026-08-02", _normal_night_timeline())
        assert 0.8 < f["sleep_efficiency"] <= 1.0
        assert f["sol_min"] == 20.0            # 22:30 上床 → 22:50 入睡
        assert f["waso_min"] == 15.0           # 01:30-01:45 清醒
        assert f["bed_exit_count"] == 1.0      # 一次 UNKNOWN 段（01:45-01:55）
        assert 0.0 < f["deep_sleep_ratio"] < 1.0

    def test_midnight_onset_same_day_key(self):
        """★ 入睡漂到午夜后，day_key 不变、特征仍算得出（原设计会在此失效）"""
        timeline = [
            {"stage": "WAKEFUL", "ts": "2026-08-02T00:10:00"},
            {"stage": "LIGHT", "ts": "2026-08-02T00:30:00"},
            {"stage": "DEEP", "ts": "2026-08-02T02:00:00"},
            {"stage": "UNKNOWN", "ts": "2026-08-02T07:00:00"},
        ]
        f = derive_sleep_features("2026-08-02", timeline)
        assert f["sleep_efficiency"] is not None
        assert f["sleep_onset_clock"] == 270.0

    def test_no_in_bed_returns_all_none(self):
        """整夜无在床段 → 全 NaN，由上层标 missing（避免"外出过夜"被误判成睡眠恶化）"""
        f = derive_sleep_features("2026-08-02", [])
        assert all(v is None for v in f.values())

    def test_events_outside_window_ignored(self):
        """搜索区间外的记录不参与计算"""
        timeline = [{"stage": "LIGHT", "ts": "2026-08-05T23:00:00"}]
        f = derive_sleep_features("2026-08-02", timeline)
        assert all(v is None for v in f.values())

    def test_dirty_records_skipped(self):
        timeline = _normal_night_timeline() + [
            {"stage": None, "ts": "2026-08-02T03:00:00"},
            {"stage": "LIGHT", "ts": "not-a-timestamp"},
            {},
        ]
        f = derive_sleep_features("2026-08-02", timeline)
        assert f["sleep_efficiency"] is not None

    def test_night_hr_only_counts_sleep_period(self):
        hearts = [
            {"avg": 100.0, "ts": "2026-08-01T20:00:00"},   # 入睡前，不计
            {"avg": 60.0, "ts": "2026-08-02T02:00:00"},    # 睡眠期内
            {"avg": 62.0, "ts": "2026-08-02T03:00:00"},
        ]
        f = derive_sleep_features("2026-08-02", _normal_night_timeline(), hearts)
        assert 60.0 <= f["night_hr_mean"] <= 62.0

    def test_daytime_nap_after_rise(self):
        timeline = _normal_night_timeline() + [
            {"stage": "LIGHT", "ts": "2026-08-02T13:00:00"},
            {"stage": "UNKNOWN", "ts": "2026-08-02T13:40:00"},
        ]
        f = derive_sleep_features("2026-08-02", timeline)
        assert f["daytime_nap_min"] >= 39.0


class TestWhstNormalization:
    def test_produces_usable_timeline(self):
        report = {
            "inBedTime": "2026-08-01T22:40:00",
            "sleepTime": "2026-08-01T23:00:00",
            "wakeTime": "2026-08-02T06:10:00",
            "outBedTime": "2026-08-02T06:35:00",
            "lightSleepDuration": 240,
            "deepSleepDuration": 110,
        }
        timeline = normalize_whst_report(report)
        f = derive_sleep_features("2026-08-02", timeline)
        assert f["sleep_efficiency"] is not None
        assert f["sol_min"] == 20.0

    def test_missing_required_fields_returns_empty(self):
        assert normalize_whst_report({}) == []
        assert normalize_whst_report({"inBedTime": "2026-08-01T22:00:00"}) == []


class TestXiaobeikeAdapter:
    def test_mock_produces_features(self):
        features = XiaobeikeAdapter(mode="mock").extract("", "2026-08-02")
        assert set(features) <= set(SLEEP_FEATURES)
        assert features["sleep_efficiency"] > 0

    def test_mock_is_deterministic(self):
        a = XiaobeikeAdapter(mode="mock").extract("", "2026-08-02")
        b = XiaobeikeAdapter(mode="mock").extract("", "2026-08-02")
        assert a == b

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            XiaobeikeAdapter(mode="mock", backend="unknown")

    def test_live_raises_with_context(self):
        with pytest.raises(NotImplementedError, match="待决问题"):
            XiaobeikeAdapter(mode="live").extract("serial", "2026-08-02")


class TestAlarmFiltering:
    def test_safety_alarms_excluded_from_activity(self):
        """★ 跌倒不得计入活动量——否则一次跌倒会被读成『活动量上升』"""
        alarms = [
            {"alarmType": 10002, "alarmTime": 1785000000000},
            {"alarmType": 12259, "alarmTime": 1785000600000},   # 跌倒
            {"alarmType": 12257, "alarmTime": 1785000900000},   # 未活动
        ]
        assert len(filter_activity_events(alarms)) == 1

    def test_audio_alarm_excluded(self):
        """v2.1 起 GRU 管线完全不采集音频"""
        alarms = [{"alarmType": 10022, "alarmTime": 1785000000000}]
        assert filter_activity_events(alarms) == []

    def test_unknown_type_excluded(self):
        """未知类型丢弃而非默认计入——否则萤石新增类型会悄悄改变活动量口径"""
        alarms = [{"alarmType": 99999, "alarmTime": 1785000000000}]
        assert filter_activity_events(alarms) == []

    def test_all_activity_types_accepted(self):
        alarms = [
            {"alarmType": t, "alarmTime": 1785000000000 + i * 120000}
            for i, t in enumerate(ALARM_TYPES_ACTIVITY)
        ]
        assert len(filter_activity_events(alarms)) == len(ALARM_TYPES_ACTIVITY)

    def test_malformed_alarms_skipped(self):
        alarms = [{"alarmType": "abc"}, {}, {"alarmTime": None}]
        assert filter_activity_events(alarms) == []


class TestDayWindow:
    def test_uses_provided_times(self):
        rise, bed = derive_day_window(
            "2026-08-02", "2026-08-02T06:30:00", "2026-08-02T22:45:00"
        )
        assert rise.hour == 6 and bed.hour == 22

    def test_falls_back_when_sleep_track_missing(self):
        """睡眠轨掉线只影响窗口精度，不影响 day_key 归属"""
        rise, bed = derive_day_window("2026-08-02")
        assert rise.hour == 7 and bed.hour == 23

    def test_inverted_times_corrected(self):
        rise, bed = derive_day_window(
            "2026-08-02", "2026-08-02T22:00:00", "2026-08-02T07:00:00"
        )
        assert bed > rise


class TestOutOfHome:
    def test_long_gap_counted(self):
        from datetime import datetime
        window = (datetime(2026, 8, 2, 7, 0), datetime(2026, 8, 2, 23, 0))
        events = ["2026-08-02T07:05:00", "2026-08-02T12:00:00"]
        minutes, count = compute_out_of_home(events, window)
        assert minutes > 200
        assert count >= 1

    def test_dense_events_no_gap(self):
        from datetime import datetime, timedelta
        window = (datetime(2026, 8, 2, 7, 0), datetime(2026, 8, 2, 23, 0))
        events = [
            (window[0] + timedelta(minutes=10 * i)).isoformat() for i in range(97)
        ]
        minutes, count = compute_out_of_home(events, window)
        assert minutes == 0.0 and count == 0

    def test_in_bed_gap_not_counted_as_out(self):
        """★ 午睡不得被算成离家（这正是 out_of_home 依赖睡眠轨的原因）"""
        from datetime import datetime, timedelta
        window = (datetime(2026, 8, 2, 7, 0), datetime(2026, 8, 2, 23, 0))
        nap_start, nap_end = datetime(2026, 8, 2, 12, 0), datetime(2026, 8, 2, 15, 0)

        # 除午睡时段外，全天每 10 分钟都有事件（确保没有其它长静默段）
        events = []
        cursor = window[0]
        while cursor <= window[1]:
            if not (nap_start <= cursor <= nap_end):
                events.append(cursor.isoformat())
            cursor += timedelta(minutes=10)

        without_bed = compute_out_of_home(events, window)[0]
        with_bed = compute_out_of_home(
            events, window, in_bed_intervals=[(nap_start, nap_end)]
        )[0]

        assert without_bed > 150, "不给在床信息时，午睡静默会被算成离家"
        assert with_bed == 0.0, "给了在床时段后，午睡不应计入离家"


class TestSocialActivityDerivation:
    def _alarms(self, hour_counts: dict[int, int]) -> list[dict]:
        from datetime import datetime, timedelta
        base = datetime(2026, 8, 2)
        out = []
        for hour, n in hour_counts.items():
            for k in range(n):
                ts = base + timedelta(hours=hour, minutes=k * 5)
                out.append({"alarmType": 10002, "alarmTime": int(ts.timestamp() * 1000)})
        return out

    def test_returns_four_features(self):
        derived = derive_social_activity_features("2026-08-02", self._alarms({9: 5, 15: 6}))
        for key in ("activity_counts", "out_of_home_min", "rar_amplitude", "rar_iv"):
            assert key in derived

    def test_low_battery_flags_degraded(self):
        """T1C 静默漏报会被误读成『活动量下降』，必须靠电量标记区分"""
        derived = derive_social_activity_features(
            "2026-08-02", self._alarms({9: 5}), t1c_battery=15
        )
        assert derived["_device_degraded"] is True

    def test_normal_battery_not_degraded(self):
        derived = derive_social_activity_features(
            "2026-08-02", self._alarms({9: 5}), t1c_battery=80
        )
        assert derived["_device_degraded"] is False

    def test_hourly_series_length(self):
        derived = derive_social_activity_features("2026-08-02", self._alarms({9: 3}))
        assert len(derived["_hourly"]) == 24

    def test_ezviz_mock_deterministic(self):
        a = EzvizEventAdapter(mode="mock").extract("", "2026-08-02")
        b = EzvizEventAdapter(mode="mock").extract("", "2026-08-02")
        assert a == b

    def test_live_raises_with_context(self):
        with pytest.raises(NotImplementedError, match="待决问题"):
            EzvizEventAdapter(mode="live").extract("serial", "2026-08-02")


class TestCircadian:
    def test_ra_full_when_night_silent(self):
        counts = np.zeros(24)
        counts[8:18] = 10.0
        assert compute_rar_amplitude(counts) == 1.0

    def test_ra_zero_when_flat(self):
        """有活动但无昼夜差别 → RA=0（作息塌陷）"""
        assert compute_rar_amplitude(np.full(24, 5.0)) == 0.0

    def test_ra_nan_when_no_activity(self):
        """★ 全零 → NaN 而非 0：『设备离线』不能被读成『作息塌陷』"""
        assert np.isnan(compute_rar_amplitude(np.zeros(24)))

    def test_iv_nan_when_constant(self):
        assert np.isnan(compute_rar_iv(np.full(24, 5.0)))

    def test_iv_higher_when_fragmented(self):
        smooth = np.array([0, 0, 1, 3, 6, 9, 11, 12, 11, 9, 8, 7,
                           6, 6, 7, 9, 11, 12, 10, 7, 4, 2, 1, 0], dtype=float)
        fragmented = np.array([0, 10] * 12, dtype=float)
        assert compute_rar_iv(fragmented) > compute_rar_iv(smooth)

    def test_m10_l5_circular_window(self):
        """★ 低谷跨午夜时必须能找到（非环形实现会漏掉 22:00-03:00 这类窗口）"""
        counts = np.full(24, 20.0)
        for h in (22, 23, 0, 1, 2):
            counts[h] = 0.0
        counts[3] = 5.0
        counts[21] = 5.0
        _, l5, _, l5_start = compute_m10_l5(counts)
        assert l5 == 0.0
        assert l5_start == 22

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            compute_rar_amplitude(np.zeros(12))

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            compute_rar_amplitude(np.full(24, -1.0))

    def test_build_hourly_dedup(self):
        events = [f"2026-08-02T09:00:{s:02d}" for s in range(0, 50, 10)]
        counts = build_hourly_counts(events, dedup_window_sec=60)
        assert counts[9] == 1.0, "60 秒内的多次触发应合并为一次"

    def test_build_hourly_separates_hours(self):
        counts = build_hourly_counts(["2026-08-02T09:00:00", "2026-08-02T10:00:00"])
        assert counts[9] == 1.0 and counts[10] == 1.0

    def test_is_needs_two_days(self):
        assert np.isnan(compute_interdaily_stability(np.zeros((1, 24))))

    def test_is_identical_days_is_one(self):
        day = np.array([0, 0, 5, 10, 8, 3] * 4, dtype=float)
        assert abs(compute_interdaily_stability(np.tile(day, (3, 1))) - 1.0) < 1e-9


class TestCopresence:
    def test_single_frame_miss_does_not_split(self):
        """★ 30 s 中位数去抖：单帧漏检不该把一次来访切成两段"""
        counts = np.full(600, 2.0)
        counts[300] = 1.0
        result = compute_copresence_minutes(counts)
        assert result["copresence_segments"] == 1
        assert result["copresence_min"] == 10.0

    def test_brief_passing_not_counted(self):
        """擦身而过（几秒钟两个人）不算社会接触"""
        counts = np.ones(600)
        counts[300:303] = 2.0
        assert compute_copresence_minutes(counts)["copresence_min"] == 0.0

    def test_sustained_visit_counted(self):
        counts = np.ones(3600)
        counts[600:2400] = 2.0
        result = compute_copresence_minutes(counts)
        assert 28.0 <= result["copresence_min"] <= 32.0

    def test_empty_input(self):
        result = compute_copresence_minutes([])
        assert result["copresence_min"] == 0.0
        assert result["has_visitor"] is False

    def test_visitor_flag(self):
        counts = np.ones(7200)
        counts[0:3600] = 2.0
        assert compute_copresence_minutes(counts)["has_visitor"] is True

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            smooth_person_counts([-1, 2, 3])

    def test_no_padding_at_boundary(self):
        """边界窗口收窄而非 padding——来访常发生在时段边缘"""
        counts = np.concatenate([np.full(120, 2.0), np.ones(480)])
        assert compute_copresence_minutes(counts)["copresence_min"] > 1.0


class TestTvRoiFilter:
    def test_detection_inside_roi_dropped(self):
        dets = [{"bbox": (0.6, 0.2, 0.8, 0.5)}]
        assert filter_tv_roi_detections(dets, (0.5, 0.1, 0.9, 0.6)) == []

    def test_detection_outside_roi_kept(self):
        dets = [{"bbox": (0.1, 0.1, 0.2, 0.4)}]
        assert len(filter_tv_roi_detections(dets, (0.5, 0.1, 0.9, 0.6))) == 1

    def test_no_roi_keeps_all(self):
        dets = [{"bbox": (0.6, 0.2, 0.8, 0.5)}]
        assert len(filter_tv_roi_detections(dets, None)) == 1

    def test_malformed_bbox_kept(self):
        """框坏了宁可保留（漏检比误删更容易被后续校准发现）"""
        assert len(filter_tv_roi_detections([{"bbox": (1, 2)}], (0, 0, 1, 1))) == 1


class TestCloudCalibration:
    def test_full_agreement_passes(self):
        result = check_cloud_calibration([1, 2, 2, 1], [1, 2, 2, 1])
        assert result["passed"] and result["alert"] is None

    def test_low_agreement_alerts(self):
        result = check_cloud_calibration([1, 2, 2, 1, 2], [1, 2, 2, 1, 1])
        assert not result["passed"]
        assert "一致率" in result["alert"]

    def test_empty_samples_not_silently_passed(self):
        """无样本必须报警，不能当成『通过』"""
        result = check_cloud_calibration([], [])
        assert not result["passed"]
        assert result["alert"] is not None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_cloud_calibration([1, 2], [1])


class TestCameraAdapter:
    def test_mock_deterministic(self):
        a = CameraAdapter(mode="mock").extract("", "2026-08-02")
        b = CameraAdapter(mode="mock").extract("", "2026-08-02")
        assert a == b

    def test_weekend_more_copresence(self):
        """周末效应必须体现，否则测不出分池 EWMA 的收益"""
        weekend = sum(
            CameraAdapter(mode="mock").extract("", d)["copresence_min"]
            for d in ("2026-08-01", "2026-08-02", "2026-08-08", "2026-08-09")
        )
        weekday = sum(
            CameraAdapter(mode="mock").extract("", d)["copresence_min"]
            for d in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")
        )
        assert weekend > weekday

    def test_live_raises_with_context(self):
        with pytest.raises(NotImplementedError, match="待决问题"):
            CameraAdapter(mode="live").extract("serial", "2026-08-02")

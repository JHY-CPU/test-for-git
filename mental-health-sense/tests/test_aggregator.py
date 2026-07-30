"""
数据聚合器单元测试（双轨）
"""

import numpy as np
import pytest

from src.baseline.scaler_utils import (
    SLEEP_FEATURES,
    SOCIAL_FEATURES,
    TRACK_SLEEP,
    TRACK_SOCIAL,
)
from src.data_pipeline.aggregator import (
    DataInsufficientError,
    aggregate_daily_features,
    aggregate_sleep_features,
    aggregate_social_features,
    aggregate_track_features,
    diagnose_social_failure,
    get_feature_value,
)


def full_sleep_data() -> dict:
    return {
        "sleep_efficiency": 0.85,
        "waso_min": 32.0,
        "sol_min": 15.0,
        "bed_exit_count": 2.0,
        "deep_sleep_ratio": 0.22,
        "sleep_onset_clock": 150.0,
        "night_hr_mean": 61.0,
        "daytime_nap_min": 35.0,
    }


def full_activity_data() -> dict:
    return {
        "out_of_home_min": 90.0,
        "rar_amplitude": 0.91,
        "rar_iv": 0.30,
        "activity_counts": 180.0,
    }


def full_camera_data() -> dict:
    return {"copresence_min": 55.0}


class TestSubAggregators:
    """测试子维度聚合"""

    def test_sleep_all_present(self):
        result = aggregate_sleep_features(full_sleep_data())
        assert result["sleep_efficiency"] == 0.85
        assert result["waso_min"] == 32.0
        assert result["bed_exit_count"] == 2.0
        assert all(v is not None for v in result.values())

    def test_sleep_partial(self):
        result = aggregate_sleep_features({"sleep_efficiency": 0.85})
        assert result["sleep_efficiency"] == 0.85
        assert result["waso_min"] is None

    def test_sleep_none(self):
        result = aggregate_sleep_features(None)
        assert set(result) == set(SLEEP_FEATURES)
        assert all(v is None for v in result.values())

    def test_social_none(self):
        result = aggregate_social_features(None, None)
        assert set(result) == set(SOCIAL_FEATURES)
        assert all(v is None for v in result.values())

    def test_social_merges_two_sources(self):
        """activity 与 camera 两路合并，各自补齐对方缺的特征"""
        result = aggregate_social_features(full_activity_data(), full_camera_data())
        assert result["copresence_min"] == 55.0      # 来自 camera
        assert result["activity_counts"] == 180.0    # 来自 activity
        assert all(v is not None for v in result.values())

    def test_camera_not_overwritten_by_missing_activity(self):
        """activity 路缺 copresence 时不能把 camera 给的值覆盖成 None"""
        result = aggregate_social_features({"activity_counts": 100.0}, full_camera_data())
        assert result["copresence_min"] == 55.0

    def test_bool_rejected(self):
        """布尔值不能被当成 1.0/0.0 混进特征（isinstance(True, int) 为真的陷阱）"""
        result = aggregate_sleep_features({"sleep_efficiency": True})
        assert result["sleep_efficiency"] is None

    def test_non_finite_rejected(self):
        result = aggregate_sleep_features({"waso_min": float("inf")})
        assert result["waso_min"] is None


class TestAggregateTrack:
    """测试单轨组装"""

    def test_sleep_track_shape(self):
        vec = aggregate_track_features(TRACK_SLEEP, aggregate_sleep_features(full_sleep_data()))
        assert vec.shape == (8,)
        assert not np.any(np.isnan(vec))

    def test_social_track_shape(self):
        vec = aggregate_track_features(
            TRACK_SOCIAL, aggregate_social_features(full_activity_data(), full_camera_data())
        )
        assert vec.shape == (5,)
        assert not np.any(np.isnan(vec))

    def test_ordering_matches_feature_names(self):
        """向量下标顺序必须与该轨 FEATURE_NAMES 一致"""
        vec = aggregate_track_features(TRACK_SLEEP, aggregate_sleep_features(full_sleep_data()))
        assert vec[SLEEP_FEATURES.index("sleep_efficiency")] == 0.85
        assert vec[SLEEP_FEATURES.index("daytime_nap_min")] == 35.0

        vec_s = aggregate_track_features(
            TRACK_SOCIAL, aggregate_social_features(full_activity_data(), full_camera_data())
        )
        assert vec_s[SOCIAL_FEATURES.index("copresence_min")] == 55.0
        assert vec_s[SOCIAL_FEATURES.index("activity_counts")] == 180.0

    def test_partial_ok_below_threshold(self):
        """缺 2 维（<3）→ 允许通过，缺失位为 NaN"""
        data = full_sleep_data()
        del data["night_hr_mean"]
        del data["daytime_nap_min"]
        vec = aggregate_track_features(TRACK_SLEEP, aggregate_sleep_features(data))
        assert vec.shape == (8,)
        assert int(np.isnan(vec).sum()) == 2

    def test_insufficient_raises(self):
        """缺 ≥3 维 → 抛异常并带上轨道名与缺失清单"""
        data = {"sleep_efficiency": 0.85}
        with pytest.raises(DataInsufficientError) as exc:
            aggregate_track_features(TRACK_SLEEP, aggregate_sleep_features(data))
        assert exc.value.track == TRACK_SLEEP
        assert exc.value.missing_count >= 3
        assert "waso_min" in exc.value.missing_features

    def test_get_feature_value(self):
        vec = aggregate_track_features(TRACK_SLEEP, aggregate_sleep_features(full_sleep_data()))
        assert get_feature_value(vec, "sol_min", TRACK_SLEEP) == 15.0
        with pytest.raises(ValueError):
            get_feature_value(vec, "copresence_min", TRACK_SLEEP)


class TestAggregateDaily:
    """测试双轨聚合的故障隔离"""

    def test_both_tracks_present(self):
        result = aggregate_daily_features(
            day_key="2026-08-01",
            sleep_data=full_sleep_data(),
            activity_data=full_activity_data(),
            camera_data=full_camera_data(),
        )
        assert result[TRACK_SLEEP].shape == (8,)
        assert result[TRACK_SOCIAL].shape == (5,)

    def test_sleep_offline_social_survives(self):
        """★ 双轨核心价值：小贝壳掉线时社交轨照常出结果"""
        result = aggregate_daily_features(
            day_key="2026-08-01",
            sleep_data=None,
            activity_data=full_activity_data(),
            camera_data=full_camera_data(),
        )
        assert result[TRACK_SLEEP] is None
        assert result[TRACK_SOCIAL] is not None
        assert not np.any(np.isnan(result[TRACK_SOCIAL]))

    def test_social_offline_sleep_survives(self):
        result = aggregate_daily_features(
            day_key="2026-08-01",
            sleep_data=full_sleep_data(),
            activity_data=None,
            camera_data=None,
        )
        assert result[TRACK_SLEEP] is not None
        assert result[TRACK_SOCIAL] is None

    def test_both_missing_raises(self):
        with pytest.raises(DataInsufficientError):
            aggregate_daily_features(day_key="2026-08-01")


class TestSocialFailureDiagnosis:
    """成组失效诊断：5 维不独立，3 维同源于 a[h]"""

    def test_all_present_valid(self):
        values = aggregate_social_features(full_activity_data(), full_camera_data())
        d = diagnose_social_failure(values, xiaobeike_online=True)
        assert d["suspected_quality"] == "valid"
        assert d["reasons"] == []

    def test_copresence_missing_kills_track(self):
        """copresence 缺失（C6c 离线）→ 整轨 missing，因为它禁止前向填充"""
        values = aggregate_social_features(full_activity_data(), None)
        d = diagnose_social_failure(values)
        assert d["suspected_quality"] == "missing"
        assert any("copresence_min" in r for r in d["reasons"])

    def test_ah_derived_all_missing_kills_track(self):
        """a[h] 全缺 → RA/IV/activity 三维同时失效"""
        values = aggregate_social_features({"out_of_home_min": 90.0}, full_camera_data())
        d = diagnose_social_failure(values)
        assert d["suspected_quality"] == "missing"

    def test_xiaobeike_offline_degrades(self):
        """小贝壳离线 → out_of_home_min 失去『非在床』条件 → degraded"""
        values = aggregate_social_features(full_activity_data(), full_camera_data())
        d = diagnose_social_failure(values, xiaobeike_online=False)
        assert d["suspected_quality"] == "degraded"
        assert any("非在床" in r for r in d["reasons"])

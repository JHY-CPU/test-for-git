"""
缺失值处理单元测试（按轨独立）
"""

import numpy as np
import pytest

from src.baseline.scaler_utils import (
    SOCIAL_FEATURES,
    TRACK_SLEEP,
    TRACK_SOCIAL,
    get_feature_dim,
)
from src.data_pipeline.imputer import (
    check_offline_status,
    impute_missing,
    impute_sequence,
)


def sleep_vec(values) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    assert vec.shape == (8,)
    return vec


def social_vec(values) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    assert vec.shape == (5,)
    return vec


class TestImputeSleepTrack:
    def test_no_missing(self):
        current = sleep_vec([1.0] * 8)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP)
        assert missing_count == 0
        assert missing_names == []
        np.testing.assert_array_almost_equal(filled, current)

    def test_single_missing_with_prev(self):
        current = sleep_vec([1.0, np.nan] + [1.0] * 6)
        prev = sleep_vec([1.0, 2.0] + [1.0] * 6)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP, prev)
        assert missing_count == 0
        assert filled[1] == 2.0

    def test_single_missing_without_prev(self):
        current = sleep_vec([1.0, np.nan] + [1.0] * 6)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP)
        assert filled[1] == 0.0
        assert missing_count == 1
        assert missing_names == ["waso_min"]

    def test_multiple_missing_partial_fill(self):
        current = sleep_vec([np.nan, 1.0, np.nan, np.nan] + [1.0] * 4)
        prev = sleep_vec([2.0, 1.0, 3.0, np.nan] + [1.0] * 4)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP, prev)
        assert filled[0] == 2.0
        assert filled[2] == 3.0
        assert filled[3] == 0.0
        assert missing_count == 1
        assert missing_names == ["bed_exit_count"]

    def test_all_missing_forward_filled(self):
        current = sleep_vec([np.nan] * 8)
        prev = sleep_vec(list(range(1, 9)))
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP, prev)
        assert missing_count == 0
        for i in range(8):
            assert filled[i] == i + 1

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="expects shape"):
            impute_missing(social_vec([1.0] * 5), TRACK_SLEEP)


class TestImputeSocialTrack:
    def test_copresence_never_forward_filled(self):
        """★ copresence_min 禁止前向填充。

        其它特征反映老人自身的行为习惯，昨天的值对今天有预测力；
        而"今天有没有人来"取决于子女的安排，用昨天填今天等于凭空伪造社会接触。
        """
        idx = SOCIAL_FEATURES.index("copresence_min")
        current = social_vec([np.nan, 90.0, 0.9, 0.3, 180.0])
        prev = social_vec([120.0, 90.0, 0.9, 0.3, 180.0])

        filled, missing_count, missing_names = impute_missing(current, TRACK_SOCIAL, prev)

        assert filled[idx] == 0.0, "不得用昨日的 120 分钟填充"
        assert missing_count == 1
        assert "copresence_min" in missing_names

    def test_other_social_features_do_forward_fill(self):
        """同一轨的其它维仍正常前向填充"""
        idx = SOCIAL_FEATURES.index("out_of_home_min")
        current = social_vec([50.0, np.nan, 0.9, 0.3, 180.0])
        prev = social_vec([50.0, 95.0, 0.9, 0.3, 180.0])

        filled, missing_count, missing_names = impute_missing(current, TRACK_SOCIAL, prev)

        assert filled[idx] == 95.0
        assert missing_count == 0

    def test_copresence_present_not_flagged(self):
        current = social_vec([50.0, 90.0, 0.9, 0.3, 180.0])
        filled, missing_count, missing_names = impute_missing(current, TRACK_SOCIAL)
        assert missing_count == 0
        assert missing_names == []


class TestImputeSequence:
    def test_short_gap_forward_filled(self):
        seq = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [np.nan] * 5,
            [1.5, 2.5, 3.5, 4.5, 5.5],
        ], dtype=np.float64)
        filled, degraded = impute_sequence(seq, TRACK_SOCIAL, max_forward_days=3)
        assert not np.any(np.isnan(filled))
        np.testing.assert_array_almost_equal(filled[1], seq[0])

    def test_long_gap_interpolated(self):
        n = 8
        seq = np.full((n, 5), np.nan, dtype=np.float64)
        seq[0] = 0.0
        seq[-1] = 7.0
        filled, degraded = impute_sequence(seq, TRACK_SOCIAL, max_forward_days=2)
        assert not np.any(np.isnan(filled))
        # 线性插值：中间值应递增
        assert filled[1, 0] < filled[3, 0] < filled[5, 0]
        assert degraded["copresence_min"] > 0

    def test_leading_gap_zero_filled(self):
        seq = np.full((4, 8), 1.0, dtype=np.float64)
        seq[0] = np.nan
        filled, degraded = impute_sequence(seq, TRACK_SLEEP)
        assert np.all(filled[0] == 0.0)

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="expects"):
            impute_sequence(np.zeros((5, 5)), TRACK_SLEEP)


class TestTrackDims:
    def test_dims_are_eight_and_five(self):
        assert get_feature_dim(TRACK_SLEEP) == 8
        assert get_feature_dim(TRACK_SOCIAL) == 5


class TestOfflineCheck:
    def test_no_offline(self):
        quality = ["valid", "valid", "insufficient", "valid", "valid"]
        assert not check_offline_status(quality, threshold=3)

    def test_offline_detected(self):
        quality = ["valid", "insufficient", "insufficient", "insufficient"]
        assert check_offline_status(quality, threshold=3)

    def test_offline_with_offline_marker(self):
        quality = ["valid", "insufficient", "offline", "offline"]
        assert check_offline_status(quality, threshold=3)

    def test_not_enough_data(self):
        quality = ["insufficient", "insufficient"]
        assert not check_offline_status(quality, threshold=3)

    def test_threshold_custom(self):
        quality = ["insufficient", "insufficient"]
        assert check_offline_status(quality, threshold=2)

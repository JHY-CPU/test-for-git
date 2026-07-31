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
    impute_missing,
)
from src.data_pipeline.validator import check_prolonged_degradation


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
        """★ 填不上的维保持 NaN，绝不填 0。

        filled 是**原始量纲**向量，会原样写进 features_{track}.csv。
        在这里填 0 意味着"老人的 WASO 是 0 分钟"，而实际是"没测到"。
        实测 E001：night_hr_mean 原始 0 → z = −15.2，sleep_efficiency → z = −11.8，
        这种假偏离会进 StandardScaler.fit 并永久带偏归一化基准。
        NaN 才是诚实的表示，推理层据此把该维排除出打分。
        """
        current = sleep_vec([1.0, np.nan] + [1.0] * 6)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP)
        assert np.isnan(filled[1])
        assert missing_count == 1
        assert missing_names == ["waso_min"]

    def test_multiple_missing_partial_fill(self):
        current = sleep_vec([np.nan, 1.0, np.nan, np.nan] + [1.0] * 4)
        prev = sleep_vec([2.0, 1.0, 3.0, np.nan] + [1.0] * 4)
        filled, missing_count, missing_names = impute_missing(current, TRACK_SLEEP, prev)
        assert filled[0] == 2.0
        assert filled[2] == 3.0
        assert np.isnan(filled[3])   # 昨天也缺 → 填不上 → 保持 NaN
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

        # NaN 而不是 0：0 的语义是"确实一个人都没来"，而摄像头掉线时我们
        # 根本不知道。copresence_min 是 social_decline 的必选维，填 0 会产生
        # 一个满足 down 方向的假 z，把"测不到"变成"确实没人来"。
        assert np.isnan(filled[idx]), "不得用昨日的 120 分钟填充，也不得填 0"
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


class TestTrackDims:
    def test_dims_are_eight_and_five(self):
        assert get_feature_dim(TRACK_SLEEP) == 8
        assert get_feature_dim(TRACK_SOCIAL) == 5


class TestOfflineCheck:
    """离线判据统一由 validator 提供。

    imputer.check_offline_status 曾是同一逻辑的第二份实现（只有默认值 3 vs 5
    不同）。两份会漂的"离线判据"比没有更危险：改了一处忘另一处，界面与告警
    就会给出互相矛盾的结论。已删除，测试改指向唯一的那份。
    """

    def test_no_offline(self):
        quality = ["valid", "valid", "insufficient", "valid", "valid"]
        assert not check_prolonged_degradation(quality, threshold=3)

    def test_offline_detected(self):
        quality = ["valid", "insufficient", "insufficient", "insufficient"]
        assert check_prolonged_degradation(quality, threshold=3)

    def test_offline_with_offline_marker(self):
        quality = ["valid", "insufficient", "offline", "offline"]
        assert check_prolonged_degradation(quality, threshold=3)

    def test_not_enough_data(self):
        quality = ["insufficient", "insufficient"]
        assert not check_prolonged_degradation(quality, threshold=3)

    def test_threshold_custom(self):
        quality = ["insufficient", "insufficient"]
        assert check_prolonged_degradation(quality, threshold=2)

    def test_degraded_also_counts_as_prolonged(self):
        """degraded 也算"非 valid"——连续降级同样意味着失去监测能力"""
        quality = ["valid", "degraded", "degraded", "degraded"]
        assert check_prolonged_degradation(quality, threshold=3)

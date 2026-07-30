"""
数据校验单元测试（双轨，四态质量）
"""

import numpy as np
import pytest

from src.baseline.scaler_utils import TRACK_SLEEP, TRACK_SOCIAL
from src.data_pipeline.validator import (
    QUALITY_DEGRADED,
    QUALITY_INSUFFICIENT,
    QUALITY_OFFLINE,
    QUALITY_VALID,
    check_prolonged_degradation,
    counts_toward_consecutive,
    describe_track_capability,
    get_quality_summary,
    is_usable_for_inference,
    is_usable_for_training,
    missing_critical_features,
    validate_daily_data,
)


def sleep_vec(fill: float = 1.0) -> np.ndarray:
    return np.full(8, fill, dtype=np.float64)


def social_vec(fill: float = 1.0) -> np.ndarray:
    return np.full(5, fill, dtype=np.float64)


class TestValidateDailyData:
    def test_valid_data_both_tracks(self):
        assert validate_daily_data(sleep_vec(), 0, TRACK_SLEEP) == QUALITY_VALID
        assert validate_daily_data(social_vec(), 0, TRACK_SOCIAL) == QUALITY_VALID

    def test_one_missing_still_valid(self):
        """缺 1 维前向填充后仍算 valid"""
        assert validate_daily_data(sleep_vec(), 1, TRACK_SLEEP) == QUALITY_VALID

    def test_two_missing_degraded(self):
        """★ 缺 2 维 → degraded：照常进模型算分，但不计入连续偏离天数"""
        assert validate_daily_data(sleep_vec(), 2, TRACK_SLEEP) == QUALITY_DEGRADED

    def test_three_missing_insufficient(self):
        assert validate_daily_data(sleep_vec(), 3, TRACK_SLEEP) == QUALITY_INSUFFICIENT
        assert validate_daily_data(social_vec(), 3, TRACK_SOCIAL) == QUALITY_INSUFFICIENT

    def test_device_degraded_flag(self):
        """外部降级信号（如 T1C 电量<20%）即使不缺维也应降级"""
        assert validate_daily_data(
            social_vec(), 0, TRACK_SOCIAL, device_degraded=True
        ) == QUALITY_DEGRADED

    def test_offline_after_continuous_insufficient(self):
        recent = [QUALITY_INSUFFICIENT] * 3
        assert validate_daily_data(
            sleep_vec(), 4, TRACK_SLEEP, recent_quality=recent
        ) == QUALITY_OFFLINE

    def test_offline_with_mixed_history(self):
        recent = [QUALITY_INSUFFICIENT, QUALITY_OFFLINE, QUALITY_OFFLINE]
        assert validate_daily_data(
            sleep_vec(), 3, TRACK_SLEEP, recent_quality=recent
        ) == QUALITY_OFFLINE

    def test_not_offline_with_interruption(self):
        recent = [QUALITY_INSUFFICIENT, QUALITY_VALID, QUALITY_INSUFFICIENT]
        assert validate_daily_data(
            sleep_vec(), 3, TRACK_SLEEP, recent_quality=recent
        ) != QUALITY_OFFLINE

    def test_extreme_outlier(self):
        vec = sleep_vec()
        vec[0] = -999.0
        assert validate_daily_data(vec, 0, TRACK_SLEEP) == QUALITY_INSUFFICIENT

    def test_wrong_shape_raises(self):
        """维度不匹配必须报错而不是静默接受——错维度会让整条残差链失去意义"""
        with pytest.raises(ValueError, match="expects shape"):
            validate_daily_data(social_vec(), 0, TRACK_SLEEP)
        with pytest.raises(ValueError, match="expects shape"):
            validate_daily_data(sleep_vec(), 0, TRACK_SOCIAL)


class TestCriticalFeatures:
    """★ copresence_min 是社会连接轨唯一的社会接触指标，缺了就不能宣称测到社会连接"""

    def test_copresence_missing_forces_degraded(self):
        quality = validate_daily_data(
            social_vec(), 1, TRACK_SOCIAL, missing_features=["copresence_min"]
        )
        assert quality == QUALITY_DEGRADED, (
            "只缺 1 维本来算 valid，但 copresence 是关键维，必须降级——"
            "否则系统会拿 4 个活动/节律维继续输出『社会连接正常』"
        )

    def test_non_critical_single_missing_stays_valid(self):
        quality = validate_daily_data(
            social_vec(), 1, TRACK_SOCIAL, missing_features=["out_of_home_min"]
        )
        assert quality == QUALITY_VALID

    def test_sleep_has_no_critical_feature(self):
        """睡眠轨没有单一不可替代维，按缺失个数判即可"""
        quality = validate_daily_data(
            sleep_vec(), 1, TRACK_SLEEP, missing_features=["sleep_efficiency"]
        )
        assert quality == QUALITY_VALID

    def test_missing_critical_helper(self):
        assert missing_critical_features(TRACK_SOCIAL, ["copresence_min", "rar_iv"]) == \
               ["copresence_min"]
        assert missing_critical_features(TRACK_SOCIAL, ["rar_iv"]) == []
        assert missing_critical_features(TRACK_SLEEP, ["waso_min"]) == []

    def test_capability_note_explains_limitation(self):
        """对外必须明示『测不到』，不能沉默"""
        note = describe_track_capability(TRACK_SOCIAL, ["copresence_min"])
        assert note is not None
        assert "社会接触" in note
        assert describe_track_capability(TRACK_SOCIAL, ["rar_iv"]) is None

    def test_backward_compatible_without_missing_features(self):
        """不传 missing_features 时行为不变（旧调用点仍可用）"""
        assert validate_daily_data(social_vec(), 1, TRACK_SOCIAL) == QUALITY_VALID


class TestUsability:
    def test_valid_for_training(self):
        assert is_usable_for_training(QUALITY_VALID)

    def test_degraded_not_for_training(self):
        """建基线只用最干净的数据"""
        assert not is_usable_for_training(QUALITY_DEGRADED)

    def test_degraded_still_usable_for_inference(self):
        """★ degraded 仍出结果——直接作废太浪费，只是不攒连续天数"""
        assert is_usable_for_inference(QUALITY_DEGRADED)

    def test_insufficient_not_usable(self):
        assert not is_usable_for_training(QUALITY_INSUFFICIENT)
        assert not is_usable_for_inference(QUALITY_INSUFFICIENT)
        assert not is_usable_for_inference(QUALITY_OFFLINE)

    def test_only_valid_counts_toward_consecutive(self):
        assert counts_toward_consecutive(QUALITY_VALID)
        assert not counts_toward_consecutive(QUALITY_DEGRADED)
        assert not counts_toward_consecutive(QUALITY_INSUFFICIENT)
        assert not counts_toward_consecutive(QUALITY_OFFLINE)


class TestQualitySummary:
    def test_all_valid(self):
        summary = get_quality_summary([QUALITY_VALID] * 5)
        assert summary["valid_days"] == 5
        assert summary["degraded_days"] == 0
        assert summary["valid_ratio"] == 1.0

    def test_mixed(self):
        summary = get_quality_summary([
            QUALITY_VALID, QUALITY_INSUFFICIENT, QUALITY_VALID,
            QUALITY_OFFLINE, QUALITY_DEGRADED,
        ])
        assert summary["valid_days"] == 2
        assert summary["insufficient_days"] == 1
        assert summary["offline_days"] == 1
        assert summary["degraded_days"] == 1
        assert summary["total_days"] == 5
        assert summary["valid_ratio"] == 0.4

    def test_empty(self):
        summary = get_quality_summary([])
        assert summary["total_days"] == 0
        assert summary["valid_ratio"] == 0.0


class TestProlongedDegradation:
    """长期降级必须与长期正常可区分——传感器坏了 5 天，界面不该显示『一切正常』"""

    def test_five_consecutive_non_valid_triggers(self):
        assert check_prolonged_degradation([QUALITY_DEGRADED] * 5, threshold=5)
        assert check_prolonged_degradation([QUALITY_INSUFFICIENT] * 5, threshold=5)

    def test_mixed_non_valid_triggers(self):
        history = [
            QUALITY_DEGRADED, QUALITY_INSUFFICIENT, QUALITY_DEGRADED,
            QUALITY_OFFLINE, QUALITY_DEGRADED,
        ]
        assert check_prolonged_degradation(history, threshold=5)

    def test_one_valid_day_resets(self):
        history = [QUALITY_DEGRADED] * 4 + [QUALITY_VALID]
        assert not check_prolonged_degradation(history, threshold=5)

    def test_too_short_history(self):
        assert not check_prolonged_degradation([QUALITY_DEGRADED] * 3, threshold=5)

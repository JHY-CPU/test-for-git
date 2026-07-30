"""
训练数据健康门禁 + 冷启动兜底单元测试（按轨）
"""

import numpy as np
import pytest

from src.baseline.cold_start_fallback import fallback_deviation_check
from src.baseline.data_health import detect_outlier_days, describe_outlier_days
from src.baseline.scaler_utils import TRACK_SLEEP, TRACK_SOCIAL

# 睡眠轨 8 维的典型基线与波动幅度
# 顺序：SE, WASO, SOL, bed_exit, deep_ratio, onset_clock, night_hr, nap
SLEEP_BASE = np.array([0.88, 35.0, 18.0, 1.5, 0.22, 150.0, 62.0, 40.0])
SLEEP_SCALE = np.array([0.02, 5.0, 4.0, 0.5, 0.02, 20.0, 3.0, 10.0])

# 社交轨 5 维：copresence, out_of_home, RA, IV, activity
SOCIAL_BASE = np.array([55.0, 95.0, 0.91, 0.30, 185.0])
SOCIAL_SCALE = np.array([12.0, 20.0, 0.02, 0.05, 15.0])


def normal_block(track: str, n_days: int = 14, seed: int = 0) -> np.ndarray:
    """生成一段平稳的正常数据"""
    rng = np.random.RandomState(seed)
    base, scale = (
        (SLEEP_BASE, SLEEP_SCALE) if track == TRACK_SLEEP else (SOCIAL_BASE, SOCIAL_SCALE)
    )
    return base + rng.normal(0, 1, (n_days, len(base))) * scale


class TestDetectOutlierDays:
    """MAD 离群检测算法本身与维度无关，两轨都要能用"""

    def test_clean_sleep_data_no_outliers(self):
        report = detect_outlier_days(normal_block(TRACK_SLEEP, seed=1))
        assert report["outlier_day_indices"] == []
        assert report["outlier_ratio"] == 0.0
        assert report["n_days"] == 14

    def test_clean_social_data_no_outliers(self):
        report = detect_outlier_days(normal_block(TRACK_SOCIAL, seed=1))
        assert report["outlier_day_indices"] == []

    def test_injected_outlier_day_detected(self):
        data = normal_block(TRACK_SLEEP, seed=2)
        data[5, 0] = 0.2     # SE 骤降
        data[5, 1] = 200.0   # WASO 飙升
        data[5, 3] = 12.0    # 离床次数飙升
        report = detect_outlier_days(data, z_threshold=3.5, min_bad_features=2)
        assert 5 in report["outlier_day_indices"]

    def test_single_feature_spike_below_min_bad(self):
        data = normal_block(TRACK_SLEEP, seed=3)
        data[7, 6] = 150.0   # 仅夜间心率异常
        report = detect_outlier_days(data, z_threshold=3.5, min_bad_features=2)
        assert 7 not in report["outlier_day_indices"]

    def test_constant_feature_no_divide_by_zero(self):
        data = normal_block(TRACK_SLEEP, seed=4)
        data[:, 2] = 18.0    # SOL 完全恒定
        report = detect_outlier_days(data)
        assert not report["feature_flags"][:, 2].any()

    def test_describe_uses_track_feature_names(self):
        """★ 描述函数按轨取特征名，两轨的输出不能混"""
        data = normal_block(TRACK_SLEEP, seed=5)
        data[3, 0] = 0.2
        data[3, 1] = 200.0
        report = detect_outlier_days(data, min_bad_features=2)
        lines = describe_outlier_days(data, report, TRACK_SLEEP)
        assert any("Day#3" in ln for ln in lines)
        assert any("sleep_efficiency" in ln for ln in lines)
        assert any("[sleep]" in ln for ln in lines)

    def test_describe_social_names(self):
        data = normal_block(TRACK_SOCIAL, seed=6)
        data[2, 0] = 500.0
        data[2, 4] = 900.0
        report = detect_outlier_days(data, min_bad_features=2)
        lines = describe_outlier_days(data, report, TRACK_SOCIAL)
        assert any("copresence_min" in ln for ln in lines)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            detect_outlier_days(np.zeros(10))


class TestColdStartFallback:
    def test_normal_today_no_deviation(self):
        history = normal_block(TRACK_SLEEP, n_days=10, seed=6)
        today = np.median(history, axis=0)
        weights = np.ones(8)
        out = fallback_deviation_check(history, today, weights, TRACK_SLEEP, sigma=3.0)
        assert out["is_deviation"] is False
        assert out["method"] == "cold_start_fallback"

    def test_extreme_today_flagged(self):
        history = normal_block(TRACK_SLEEP, n_days=10, seed=7)
        today = np.median(history, axis=0).copy()
        today[0] -= 20 * history[:, 0].std()
        today[1] += 20 * history[:, 1].std()
        weights = np.ones(8)
        out = fallback_deviation_check(history, today, weights, TRACK_SLEEP, sigma=3.0)
        assert out["is_deviation"] is True

    def test_insufficient_history_skipped(self):
        """历史样本 <3 的特征不参与判定，不应崩溃"""
        history = normal_block(TRACK_SLEEP, n_days=2, seed=8)
        today = history[0]
        weights = np.ones(8)
        out = fallback_deviation_check(history, today, weights, TRACK_SLEEP, sigma=3.0)
        assert out["anomaly_score"] == 0.0
        assert len(out["skipped_features"]) == 8

    def test_signed_direction_preserved(self):
        """★ 修掉 np.abs 的收益：兜底期也能判方向"""
        history = normal_block(TRACK_SOCIAL, n_days=10, seed=9)
        today = np.median(history, axis=0).copy()
        today[0] = 2.0      # copresence 骤降
        today[3] = 2.5      # IV 飙升
        weights = np.ones(5)
        out = fallback_deviation_check(history, today, weights, TRACK_SOCIAL, sigma=3.0)
        assert out["feature_z"]["copresence_min"] < 0, "下降应为负"
        assert out["feature_z"]["rar_iv"] > 0, "上升应为正"
        assert out["feature_z_abs"]["copresence_min"] > 0, "abs 恒为正"

    def test_missing_dims_do_not_dilute_score(self):
        """★ 缺陷1：缺失维不得稀释异常分（分母只算有效维）"""
        history = normal_block(TRACK_SOCIAL, n_days=12, seed=10)
        med = np.median(history, axis=0)

        today_full = med.copy()
        today_full[0] = 2.0
        today_full[1] = 5.0
        today_full[4] = 30.0
        weights = np.ones(5)
        full = fallback_deviation_check(history, today_full, weights, TRACK_SOCIAL, sigma=3.0)

        # 同样的偏离，但另外两维今日缺失
        today_partial = today_full.copy()
        today_partial[2] = np.nan
        today_partial[3] = np.nan
        partial = fallback_deviation_check(
            history, today_partial, weights, TRACK_SOCIAL, sigma=3.0
        )

        assert full["is_deviation"], "全维偏离应检出"
        assert partial["is_deviation"], "缺 2 维后仍应检出（不得被稀释成正常）"
        assert set(partial["skipped_features"]) == {"rar_amplitude", "rar_iv"}

    def test_constant_history_sudden_jump_detected(self):
        """★ 缺陷2：过去恒定、今天突变必须检出（旧版直接 continue 跳过）"""
        history = np.tile(SOCIAL_BASE, (12, 1))
        today = SOCIAL_BASE.copy()
        today[4] = 20.0     # activity_counts 从恒定 185 突变到 20
        weights = np.ones(5)
        out = fallback_deviation_check(history, today, weights, TRACK_SOCIAL, sigma=3.0)
        assert out["is_deviation"], "恒定历史下的突变不应被跳过"
        assert out["feature_z"]["activity_counts"] < 0

    def test_wrong_track_dim_raises(self):
        history = normal_block(TRACK_SOCIAL, n_days=10, seed=11)
        today = np.median(history, axis=0)
        with pytest.raises(ValueError):
            fallback_deviation_check(history, today, np.ones(5), TRACK_SLEEP, sigma=3.0)

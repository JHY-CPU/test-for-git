"""UnifiedDataManager 与实时聚合的时间同步 / 自然日聚合 / 缺失标记回归测试。

覆盖待修清单 P1-1（滑动窗口墙上时钟过滤、按自然日聚合）与 P1-2（缺失标记）。
"""

import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.realtime.sensevoice_engine import (
    RealtimeFeatureAggregator,
    aggregate_acoustic_from_utterances,
    NEUTRAL_ACOUSTIC,
)
from src.unified_data_manager import UnifiedDataManager


def _utt(emotion="neutral", speed=4.0, pitch=200.0, dur=3.0, start=0.0):
    return {
        "start_sec": start,
        "duration_sec": dur,
        "emotion": emotion,
        "speech_rate": speed,
        "pitch_mean": pitch,
    }


class TestSharedAggregation:
    def test_empty_returns_neutral(self):
        assert aggregate_acoustic_from_utterances([]) == NEUTRAL_ACOUSTIC

    def test_sad_ratio_and_distress(self):
        utts = [_utt("sad", dur=2.0), _utt("neutral", dur=2.0)]
        feats = aggregate_acoustic_from_utterances(utts)
        assert feats["sad_ratio"] == pytest.approx(0.5)
        assert feats["distress_events"] == 1
        assert feats["n_utterances"] == 2


class TestWallClockWindow:
    def test_stale_utterance_excluded_at_read(self):
        """窗口冻结场景：老数据（>24h）在读取时按墙上时钟被排除，不再算进当前特征。"""
        agg = RealtimeFeatureAggregator(window_hours=24)
        now = time.time()
        # 一条 25 小时前的旧数据 + 一条刚刚的新数据
        agg.utterances_buffer = [
            (now - 25 * 3600, _utt("sad", dur=5.0)),
            (now - 60, _utt("neutral", dur=5.0)),
        ]
        feats = agg.get_current_features()
        # 只应算进新的那条 neutral，sad_ratio=0
        assert feats["n_utterances"] == 1
        assert feats["sad_ratio"] == pytest.approx(0.0)

    def test_all_stale_returns_neutral(self):
        agg = RealtimeFeatureAggregator(window_hours=24)
        now = time.time()
        agg.utterances_buffer = [(now - 30 * 3600, _utt("sad"))]
        feats = agg.get_current_features()
        assert feats["n_utterances"] == 0


class TestNaturalDayAggregation:
    def test_persist_and_aggregate_by_calendar_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UnifiedDataManager(elder_id="TEST", data_dir=tmp)
            today = datetime.now().strftime("%Y-%m-%d")
            # 今天两条：1 sad + 1 neutral
            mgr.add_realtime_utterances([_utt("sad", dur=2.0), _utt("neutral", dur=2.0)], time.time())

            day = mgr.aggregate_natural_day(today)
            assert day["data_quality"] == "valid"
            assert day["n_utterances"] == 2
            assert day["sad_ratio"] == pytest.approx(0.5)

    def test_utterances_split_across_days(self):
        """跨自然日的 utterance 归属到各自日期，互不混入。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UnifiedDataManager(elder_id="TEST", data_dir=tmp)
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            ts_today = now.timestamp()
            ts_yest = yesterday.timestamp()

            mgr.add_realtime_utterances([_utt("sad")], ts_yest)
            mgr.add_realtime_utterances([_utt("neutral")], ts_today)

            d_today = mgr.aggregate_natural_day(now.strftime("%Y-%m-%d"))
            d_yest = mgr.aggregate_natural_day(yesterday.strftime("%Y-%m-%d"))
            assert d_today["n_utterances"] == 1
            assert d_yest["n_utterances"] == 1
            assert d_yest["sad_ratio"] == pytest.approx(1.0)


class TestMissingMarking:
    def test_no_data_marked_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UnifiedDataManager(elder_id="TEST", data_dir=tmp)
            res = mgr.get_daily_acoustic_with_quality("2020-01-01")
            assert res["data_quality"] == "missing"
            # 返回中性默认值而非伪造偏离
            assert res["sad_ratio"] == NEUTRAL_ACOUSTIC["sad_ratio"]

    def test_present_data_marked_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UnifiedDataManager(elder_id="TEST", data_dir=tmp)
            today = datetime.now().strftime("%Y-%m-%d")
            mgr.add_realtime_utterances([_utt("neutral")], time.time())
            res = mgr.get_daily_acoustic_with_quality(today)
            assert res["data_quality"] == "valid"

    def test_corrupt_jsonl_line_skipped(self):
        """半截损坏行不拖垮整日聚合（模拟进程中断）。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UnifiedDataManager(elder_id="TEST", data_dir=tmp)
            today = datetime.now().strftime("%Y-%m-%d")
            mgr.add_realtime_utterances([_utt("neutral")], time.time())
            # 追加一行损坏 JSON
            day_file = mgr.utterances_dir / f"{today}.jsonl"
            with open(day_file, "a", encoding="utf-8") as f:
                f.write('{"ts": 123, "emotion": "sad"\n')  # 缺右括号
            day = mgr.aggregate_natural_day(today)
            assert day["n_utterances"] == 1  # 只算有效那条


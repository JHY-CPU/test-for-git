"""
统一数据流管理器 - 整合实时监测与每日推理

核心理念：
    实时系统不再是独立的监测器，而是整个系统的"数据采集前端"
    每日系统从实时系统读取累积数据，形成统一的数据流
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from src.realtime.sensevoice_engine import (
    RealtimeFeatureAggregator,
    aggregate_acoustic_from_utterances,
    NEUTRAL_ACOUSTIC,
)

# 声学 4 维（与 aggregator/每日轨对齐）
_ACOUSTIC_KEYS = ("sad_ratio", "avg_speed", "pitch_variability", "distress_events")


class UnifiedDataManager:
    """统一数据流管理器

    职责：
        1. 管理实时系统的特征聚合器（实时展示用，24h 滑动窗口）
        2. 按"自然日"持久化原始 utterance，供每日轨按自然日聚合声学特征
        3. 为每日系统提供带质量标记的 acoustic_data

    时间语义（关键）：实时展示用滑动窗口（"截至此刻最近 24h"），每日轨用自然日
    （"某年某月某日 00:00–23:59"）。二者语义不同，故分开存取，避免"滑动窗口 vs
    墙上时钟"错位——原 snapshot 方案在老人夜间静默 + 每日轨凌晨触发时会读到错位/陈旧数据。
    """

    def __init__(self, elder_id: str, data_dir: str = "./data"):
        self.elder_id = elder_id
        self.data_dir = Path(data_dir)

        # 实时特征聚合器（24小时滑动窗口，仅用于实时展示）
        self.realtime_aggregator = RealtimeFeatureAggregator(window_hours=24)

        # 数据持久化目录
        self.features_dir = self.data_dir / "realtime" / elder_id / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        # 按自然日存原始 utterance（每日轨聚合的权威来源）
        self.utterances_dir = self.data_dir / "realtime" / elder_id / "utterances"
        self.utterances_dir.mkdir(parents=True, exist_ok=True)

    def add_realtime_utterances(self, utterances: list[dict], timestamp: float):
        """
        添加实时语音片段（由实时采集模块调用）

        Args:
            utterances: SenseVoice推理结果
            timestamp: 音频块起始时间戳（Unix 秒，墙上时钟）
        """
        self.realtime_aggregator.add_utterances(utterances, timestamp)

        # 按每条 utterance 的墙上时钟归属到对应自然日，落 JSONL（append-only，抗中断）
        self._persist_utterances_by_day(utterances, timestamp)

        # 兼容旧接口：仍保存滑动窗口快照（实时展示 / 调试用）
        self._save_snapshot()

    def _persist_utterances_by_day(self, utterances: list[dict], timestamp: float):
        """把 utterance 按墙上时钟归属的自然日追加到 {date}.jsonl。"""
        for utt in utterances:
            utt_ts = timestamp + utt.get("start_sec", 0)
            day = datetime.fromtimestamp(utt_ts).strftime("%Y-%m-%d")
            record = {"ts": utt_ts, **utt}
            day_file = self.utterances_dir / f"{day}.jsonl"
            # append-only 单行写：即使中断也只可能丢最后一行，不损坏历史
            with open(day_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def aggregate_natural_day(self, date: str) -> dict:
        """聚合某自然日（00:00–23:59）的全部 utterance → 声学特征 + 质量标记。

        Returns:
            {**4维声学, "n_utterances", "total_duration", "data_quality"}
            data_quality: "valid"（有语音）/ "missing"（该日无任何 utterance）
        """
        day_file = self.utterances_dir / f"{date}.jsonl"
        utterances: list[dict] = []
        if day_file.exists():
            with open(day_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        utterances.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 跳过中断产生的半截行，不让单行损坏拖垮整日聚合
                        continue

        feats = aggregate_acoustic_from_utterances(utterances)
        feats["data_quality"] = "valid" if utterances else "missing"
        return feats

    def get_current_features(self) -> dict:
        """获取当前24小时特征（实时展示用，滑动窗口）"""
        return self.realtime_aggregator.get_current_features()

    def get_daily_acoustic_data(self, date: str) -> dict:
        """获取指定日期的声学特征（每日系统使用），仅 4 维声学值。

        统一走"自然日聚合"这一权威来源，不再对"今天/过去"用两套数据源。
        无数据时返回中性默认值（质量标记见 get_daily_acoustic_with_quality）。
        """
        return {k: self.get_daily_acoustic_with_quality(date)[k] for k in _ACOUSTIC_KEYS}

    def get_daily_acoustic_with_quality(self, date: str) -> dict:
        """带质量标记的声学特征。

        Returns:
            {**4维声学, "data_quality": "valid"|"missing"}

        data_quality="missing" 表示该自然日无任何 utterance：返回的是中性默认值而非
        真实测量。每日轨据此把该天声学视为缺失（交给 imputer/validator 处理），
        避免用假的"正常值"喂进 GRU 掩盖真实偏离。
        """
        day = self.aggregate_natural_day(date)

        # 该自然日无原始 utterance：尝试回退到旧快照（历史兼容），仍无则标 missing
        if day["data_quality"] == "missing":
            snapshot_file = self.features_dir / f"snapshot_{date}.json"
            if snapshot_file.exists():
                try:
                    with open(snapshot_file, "r", encoding="utf-8") as f:
                        acoustic = json.load(f).get("acoustic_data", {})
                    if acoustic:
                        return {**{k: acoustic.get(k, NEUTRAL_ACOUSTIC[k]) for k in _ACOUSTIC_KEYS},
                                "data_quality": "valid"}
                except (json.JSONDecodeError, OSError):
                    pass
            return {**{k: NEUTRAL_ACOUSTIC[k] for k in _ACOUSTIC_KEYS},
                    "data_quality": "missing"}

        return {**{k: day[k] for k in _ACOUSTIC_KEYS}, "data_quality": "valid"}

    def _save_snapshot(self):
        """保存当前状态快照"""
        today = datetime.now().strftime("%Y-%m-%d")
        features = self.realtime_aggregator.get_current_features()

        snapshot = {
            "date": today,
            "elder_id": self.elder_id,
            "timestamp": datetime.now().isoformat(),
            "acoustic_data": {
                "sad_ratio": features["sad_ratio"],
                "avg_speed": features["avg_speed"],
                "pitch_variability": features["pitch_variability"],
                "distress_events": features["distress_events"],
            },
            "statistics": {
                "n_utterances": features["n_utterances"],
                "total_duration": features["total_duration"],
            }
        }

        snapshot_file = self.features_dir / f"snapshot_{today}.json"
        # 原子写：先写临时文件再 os.replace 替换，进程中断不会留下半截损坏的 JSON
        tmp_file = snapshot_file.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, snapshot_file)

    def get_7day_acoustic_history(self, end_date: str) -> list[dict]:
        """
        获取最近7天的声学特征历史（GRU模型需要）

        Args:
            end_date: 结束日期 "YYYY-MM-DD"

        Returns:
            7天的acoustic_data列表
        """
        end = datetime.strptime(end_date, "%Y-%m-%d")
        history = []

        for i in range(7):
            date = (end - timedelta(days=6-i)).strftime("%Y-%m-%d")
            acoustic_data = self.get_daily_acoustic_data(date)
            history.append(acoustic_data)

        return history

    def export_for_daily_inference(self, date: str) -> dict:
        """
        导出用于每日推理的完整数据包

        Args:
            date: 日期 "YYYY-MM-DD"

        Returns:
            {
                "acoustic_data": {...},  # 来自实时系统
                "sleep_data": None,      # 需要其他传感器
                "activity_data": None,   # 需要其他传感器
                "social_data": None,     # 需要其他传感器
            }
        """
        return {
            "acoustic_data": self.get_daily_acoustic_data(date),
            "sleep_data": None,  # 留给其他传感器
            "activity_data": None,
            "social_data": None,
        }


# 使用示例
if __name__ == "__main__":
    # 初始化统一管理器
    manager = UnifiedDataManager(elder_id="E001")

    # 模拟实时数据流入
    import time
    mock_utterances = [
        {
            "start_sec": 0,
            "duration_sec": 3.0,
            "emotion": "sad",
            "speech_rate": 3.5,
            "pitch_mean": 190,
        }
    ]

    manager.add_realtime_utterances(mock_utterances, time.time())

    # 实时系统获取当前特征
    current = manager.get_current_features()
    print("当前24小时特征:", current)

    # 每日系统获取昨日特征
    today = datetime.now().strftime("%Y-%m-%d")
    acoustic = manager.get_daily_acoustic_data(today)
    print("每日系统用的acoustic_data:", acoustic)

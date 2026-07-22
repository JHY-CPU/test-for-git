"""
统一数据流管理器 - 整合实时监测与每日推理

核心理念：
    实时系统不再是独立的监测器，而是整个系统的"数据采集前端"
    每日系统从实时系统读取累积数据，形成统一的数据流
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from src.realtime.sensevoice_engine import RealtimeFeatureAggregator


class UnifiedDataManager:
    """统一数据流管理器

    职责：
        1. 管理实时系统的特征聚合器
        2. 为每日系统提供标准化的acoustic_data
        3. 维护数据一致性和完整性
    """

    def __init__(self, elder_id: str, data_dir: str = "./data"):
        self.elder_id = elder_id
        self.data_dir = Path(data_dir)

        # 实时特征聚合器（24小时滑动窗口）
        self.realtime_aggregator = RealtimeFeatureAggregator(window_hours=24)

        # 数据持久化目录
        self.features_dir = self.data_dir / "realtime" / elder_id / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def add_realtime_utterances(self, utterances: list[dict], timestamp: float):
        """
        添加实时语音片段（由实时采集模块调用）

        Args:
            utterances: SenseVoice推理结果
            timestamp: 时间戳
        """
        self.realtime_aggregator.add_utterances(utterances, timestamp)

        # 自动保存快照（防止系统重启丢失数据）
        self._save_snapshot()

    def get_current_features(self) -> dict:
        """获取当前24小时特征（实时监测使用）"""
        return self.realtime_aggregator.get_current_features()

    def get_daily_acoustic_data(self, date: str) -> dict:
        """
        获取指定日期的声学特征（每日系统使用）

        Args:
            date: 日期字符串 "YYYY-MM-DD"

        Returns:
            符合每日系统格式的acoustic_data
            {
                "sad_ratio": float,
                "avg_speed": float,
                "pitch_variability": float,
                "distress_events": int,
            }
        """
        # 方案1：从实时聚合器获取（如果是今天）
        today = datetime.now().strftime("%Y-%m-%d")
        if date == today:
            features = self.realtime_aggregator.get_current_features()
            return {
                "sad_ratio": features["sad_ratio"],
                "avg_speed": features["avg_speed"],
                "pitch_variability": features["pitch_variability"],
                "distress_events": features["distress_events"],
            }

        # 方案2：从历史快照读取（如果是过去）
        snapshot_file = self.features_dir / f"snapshot_{date}.json"
        if snapshot_file.exists():
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            return snapshot.get("acoustic_data", {})

        # 方案3：返回默认值
        return {
            "sad_ratio": 0.05,
            "avg_speed": 4.0,
            "pitch_variability": 25.0,
            "distress_events": 0,
        }

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
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

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

"""
传感器适配器层

每种传感器一个适配器类，负责：
    1. 从原始数据源读取数据（文件 / 消息队列 / SDK）
    2. 将原始数据加工为 aggregator 需要的特征 dict
    3. 统一接口：extract(raw_source, date) → dict

对接真实设备时，只需实现每个适配器的 _read_raw() 方法。
"""

from abc import ABC, abstractmethod
from datetime import datetime


class SensorAdapter(ABC):
    """传感器适配器基类"""

    def __init__(self, mode: str = "mock"):
        """
        Args:
            mode: "mock" → 返回模拟数据
                  "file" → 从 JSON 文件读取
                  "live"  → 从真实传感器读取（需实现 _read_raw）
        """
        if mode not in ("mock", "file", "live"):
            raise ValueError(f"mode must be 'mock'/'file'/'live', got '{mode}'")
        self.mode = mode

    @abstractmethod
    def _read_raw(self, source: str, date: str) -> dict:
        """
        从真实传感器读取原始数据 → 特征值 dict。

        对接真实设备 SDK 时，只需实现这个方法。返回值是已经映射好的特征值。

        Args:
            source: 数据来源标识（设备路径 / MQTT topic / API endpoint）
            date: 日期字符串 "YYYY-MM-DD"

        Returns:
            {feature_name: float_value, ...}，无数据的特征不出现
        """
        ...

    def _read_file(self, filepath: str) -> dict:
        """从 JSON 文件读取（mode='file' 时使用）"""
        import json
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 移除元数据字段，只保留特征值
        return {k: float(v) for k, v in data.items()
                if k not in ("timestamp", "source", "raw_path") and v is not None}

    @abstractmethod
    def _generate_mock(self, date: str) -> dict:
        """生成模拟数据（mode='mock' 时使用）"""
        ...

    def extract(self, source: str, date: str) -> dict:
        """
        提取特征值 dict。

        Args:
            source: 数据来源（mode='mock' 时可为任意值，'file' 时为 JSON 路径，
                    'live' 时为设备标识）
            date: 日期字符串

        Returns:
            {feature_name: float_value, ...}
        """
        if self.mode == "mock":
            return self._generate_mock(date)
        elif self.mode == "file":
            return self._read_file(source)
        elif self.mode == "live":
            return self._read_raw(source, date)
        return {}

    @staticmethod
    def _safe_float(value) -> float | None:
        """安全转换为 float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

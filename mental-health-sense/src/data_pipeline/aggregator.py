"""
数据聚合器：四维度传感器数据 → 12维特征向量

数据来源：
    - 睡眠雷达（非接触式）→ sleep_efficiency, deep_sleep_ratio, sfi, hrv_rmssd
    - PIR传感器 + IPC骨骼追踪 → daily_activity, space_entropy
    - 拾音设备 + 智能音箱 → social_turns, speech_duration_ratio
    - SenseVoice模型 → sad_ratio, avg_speed, avg_pitch, distress_events

输出：每日一行12维特征向量（10维健康特征 + 2维时间编码）
"""

from datetime import datetime, timedelta

import numpy as np

from src.baseline.scaler_utils import FULL_FEATURE_DIM, FULL_FEATURE_NAMES


# 数据不足异常
class DataInsufficientError(Exception):
    """当日数据不足以聚合（≥3个特征缺失）"""

    def __init__(self, missing_count: int, missing_features: list[str]):
        self.missing_count = missing_count
        self.missing_features = missing_features
        super().__init__(
            f"Data insufficient: {missing_count} features missing: {missing_features}"
        )


def _compute_time_features(date_str: str) -> tuple[float, float]:
    """
    计算时间编码特征。

    Args:
        date_str: "YYYY-MM-DD" 格式日期

    Returns:
        (day_sin, day_cos)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday
    days_in_year = 366 if dt.year % 4 == 0 else 365

    theta = 2 * np.pi * day_of_year / days_in_year
    return float(np.sin(theta)), float(np.cos(theta))


def aggregate_sleep_features(sleep_data: dict | None) -> dict:
    """
    从睡眠雷达数据提取睡眠特征。

    Args:
        sleep_data: 睡眠雷达原始数据字典，包含:
            - sleep_efficiency: 睡眠效率 [0, 1]
            - deep_sleep_ratio: 深睡占比 [0, 1]
            - sfi: 睡眠碎片化指数
            - hrv_rmssd: 心率变异性RMSSD (ms)

    Returns:
        {feature_name: value} 字典，None表示缺失
    """
    features = {
        "sleep_efficiency": None,
        "deep_sleep_ratio": None,
        "sfi": None,
        "hrv_rmssd": None,
    }

    if sleep_data is None:
        return features

    for key in features:
        if key in sleep_data and sleep_data[key] is not None:
            features[key] = float(sleep_data[key])

    return features


def aggregate_activity_features(activity_data: dict | None) -> dict:
    """
    从PIR传感器 + IPC骨骼追踪数据提取活动特征。

    Args:
        activity_data: 活动数据字典，包含:
            - daily_activity: 日间活动量（步数/活动时长综合）

    Returns:
        {feature_name: value} 字典
    """
    features = {
        "daily_activity": None,
    }

    if activity_data is None:
        return features

    for key in features:
        if key in activity_data and activity_data[key] is not None:
            features[key] = float(activity_data[key])

    return features


def aggregate_social_features(social_data: dict | None) -> dict:
    """
    从拾音设备 + 智能音箱数据提取社交特征。

    Args:
        social_data: 社交数据字典，包含:
            - social_turns: 对话轮次（每日交互次数）

    Returns:
        {feature_name: value} 字典
    """
    features = {
        "social_turns": None,
    }

    if social_data is None:
        return features

    for key in features:
        if key in social_data and social_data[key] is not None:
            features[key] = float(social_data[key])

    return features


def aggregate_acoustic_features(acoustic_data: dict | None) -> dict:
    """
    从SenseVoice模型提取声学/语义特征。

    Args:
        acoustic_data: 声学数据字典，包含:
            - sad_ratio: 悲伤标签占比
            - avg_speed: 平均语速（音节/秒）
            - avg_pitch: 平均基频F0
            - distress_events: 叹气/哭声频次

    Returns:
        {feature_name: value} 字典
    """
    features = {
        "sad_ratio": None,
        "avg_speed": None,
        "avg_pitch": None,
        "distress_events": None,
    }

    if acoustic_data is None:
        return features

    for key in features:
        if key in acoustic_data and acoustic_data[key] is not None:
            features[key] = float(acoustic_data[key])

    return features


def aggregate_daily_features(
    date_str: str,
    sleep_data: dict | None = None,
    activity_data: dict | None = None,
    social_data: dict | None = None,
    acoustic_data: dict | None = None,
) -> np.ndarray:
    """
    四维度原始数据 → 10维特征向量。

    Args:
        date_str: 日期字符串 "YYYY-MM-DD" (保留参数以保持接口兼容，但不再用于时间编码)
        sleep_data: 睡眠雷达数据
        activity_data: PIR + IPC数据
        social_data: 拾音 + 音箱数据
        acoustic_data: SenseVoice数据

    Returns:
        (10,) numpy数组，按 FULL_FEATURE_NAMES 顺序排列

    Raises:
        DataInsufficientError: 当≥3个特征缺失时
    """
    # 1. 聚合四维度特征
    sleep_feats = aggregate_sleep_features(sleep_data)
    activity_feats = aggregate_activity_features(activity_data)
    social_feats = aggregate_social_features(social_data)
    acoustic_feats = aggregate_acoustic_features(acoustic_data)

    # 2. 按 FULL_FEATURE_NAMES 顺序合并（10维健康特征）
    all_features = {}
    all_features.update(acoustic_feats)   # sad_ratio, avg_speed, avg_pitch, distress_events
    all_features.update(sleep_feats)      # sleep_efficiency, deep_sleep_ratio, sfi, hrv_rmssd
    all_features.update(activity_feats)   # daily_activity
    all_features.update(social_feats)     # social_turns

    # 3. 统计缺失
    health_names = FULL_FEATURE_NAMES  # 所有10维都是健康特征
    missing_features = [
        name for name in health_names if all_features.get(name) is None
    ]
    missing_count = len(missing_features)

    # 4. 检查数据充足性
    if missing_count >= 3:
        raise DataInsufficientError(missing_count, missing_features)

    # 5. 组装10维向量
    vector = np.zeros(FULL_FEATURE_DIM, dtype=np.float64)

    for i, name in enumerate(FULL_FEATURE_NAMES):
        if name in all_features and all_features[name] is not None:
            vector[i] = float(all_features[name])
        else:
            vector[i] = np.nan  # 待填充

    return vector


def get_feature_value(vector: np.ndarray, feature_name: str) -> float:
    """从特征向量中提取指定特征值"""
    if feature_name in FULL_FEATURE_NAMES:
        idx = FULL_FEATURE_NAMES.index(feature_name)
        return float(vector[idx])
    raise ValueError(f"Unknown feature: {feature_name}")

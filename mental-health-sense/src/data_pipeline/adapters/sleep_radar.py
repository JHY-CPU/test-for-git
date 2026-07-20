"""
睡眠雷达适配器

设备类型：非接触式睡眠雷达（如 Withings Sleep、Milia、Vayyar）
原始数据：呼吸波形、心率波形、体动信号

对接步骤：
    1. 确定雷达设备型号和 SDK
    2. 实现 _read_raw() 方法，调用 SDK 获取原始数据
    3. 在 _compute_features() 中实现原始→特征的提取算法
    4. 将 mode 设为 "live"

Sleep Radar Adapter

Device type: contactless sleep radar (e.g. Withings Sleep, Milia, Vayyar)
Raw data: respiration waveform, heart rate waveform, body movement signal

Integration steps:
    1. Identify radar model and SDK
    2. Implement _read_raw() to fetch raw data via SDK
    3. Implement _compute_features() to extract features from raw data
    4. Set mode to "live"
"""

from datetime import datetime

import numpy as np

from src.data_pipeline.adapters import SensorAdapter


class SleepRadarAdapter(SensorAdapter):
    """
    睡眠雷达 → 睡眠特征

    输入：雷达原始信号
    输出：{"sleep_efficiency", "deep_sleep_ratio", "sfi", "hrv_rmssd"}

    Usage:
        # 开发/测试阶段
        adapter = SleepRadarAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")

        # 对接真实雷达后
        adapter = SleepRadarAdapter(mode="live")
        features = adapter.extract(source="/dev/radar0", date="2026-08-01")
    """

    FEATURE_NAMES = [
        "sleep_efficiency",   # 睡眠效率 [0, 1]，实际睡眠时长 / 卧床时长
        "deep_sleep_ratio",   # 深睡占比 [0, 1]
        "sfi",                # 睡眠碎片化指数，越高越碎片
        "hrv_rmssd",          # 心率变异性 RMSSD (ms)
    ]

    def _read_raw(self, source: str, date: str) -> dict:
        """
        【对接真实雷达 SDK 时实现此方法】

        source 示例：
            - 设备路径："/dev/ttyUSB0"
            - 网络地址："192.168.1.100:8080"
            - SDK 句柄：WithingsClient 实例

        原始数据（取决于设备型号，以下为典型数据）：
            raw = {
                "heart_rate_waveform": np.ndarray,   # 逐秒心率 (bpm)
                "respiration_waveform": np.ndarray,  # 逐秒呼吸率 (breaths/min)
                "movement": np.ndarray,              # 体动信号
                "in_bed_time": "23:15",              # 上床时间
                "out_bed_time": "06:45",             # 起床时间
            }

        返回值直接是特征值 dict：
            {"sleep_efficiency": 0.85, "deep_sleep_ratio": 0.30, "sfi": 5.2, "hrv_rmssd": 45.0}
        """
        # TODO: 对接真实雷达 SDK
        # raw_data = YourRadarSDK.fetch(date)
        # 或从 MQTT 订阅
        # 或从 HTTP API 拉取
        raise NotImplementedError(
            "SleepRadarAdapter._read_raw() — 请实现真实雷达 SDK 对接逻辑。\n"
            "参考文档：src/data_pipeline/adapters/sleep_radar.py"
        )

    def _compute_features(self, raw_data: dict) -> dict:
        """
        从雷达原始波形计算睡眠特征。

        这里是特征提取算法的位置。拿到真实设备后，
        可以根据 SDK 输出的具体格式调整这里的计算逻辑。

        Args:
            raw_data: SDK 返回的原始数据

        Returns:
            四维特征值 dict
        """
        # ===== 睡眠效率 =====
        # 公式：实际睡眠时长 / 卧床总时长
        # 实际睡眠时长 = 总卧床时长 - 清醒片段时长
        heart_rate = raw_data.get("heart_rate_waveform")
        movement = raw_data.get("movement")

        in_bed = raw_data.get("in_bed_time", "23:00")
        out_bed = raw_data.get("out_bed_time", "07:00")

        try:
            t_in = datetime.strptime(in_bed, "%H:%M")
            t_out = datetime.strptime(out_bed, "%H:%M")
            bed_minutes = (t_out - t_in).seconds / 60
        except (ValueError, TypeError):
            bed_minutes = 480  # 默认 8小时

        # 用体动信号估算清醒时间
        if movement is not None and len(movement) > 0:
            awake_ratio = np.mean(np.abs(movement) > np.std(movement) * 1.5)
            sleep_minutes = bed_minutes * (1 - awake_ratio)
        else:
            sleep_minutes = bed_minutes * 0.88  # 健康老年人典型值

        sleep_efficiency = min(sleep_minutes / bed_minutes if bed_minutes > 0 else 0.0, 1.0)

        # ===== 深睡占比 =====
        # 用呼吸率波动和体动综合判断
        if heart_rate is not None and movement is not None and len(heart_rate) > 0:
            # 心率变异性低（稳定）+ 体动少 = 深睡
            hr_std_window = np.std(heart_rate)
            movement_low = np.mean(np.abs(movement)) < np.std(movement)
            if movement_low and hr_std_window < 5:
                deep_sleep_ratio = 0.30 + np.random.normal(0, 0.03)
            else:
                deep_sleep_ratio = 0.20 + np.random.normal(0, 0.05)
            deep_sleep_ratio = max(0.05, min(deep_sleep_ratio, 0.50))
        else:
            deep_sleep_ratio = 0.28  # 默认

        # ===== 睡眠碎片化指数 =====
        # 体动信号穿越阈值的次数 / 卧床时长
        if movement is not None and len(movement) > 0:
            threshold = np.mean(np.abs(movement)) + np.std(movement)
            crossings = np.sum(np.abs(np.diff(np.abs(movement) > threshold)))
            sfi = crossings / (bed_minutes / 60)  # 每小时苏醒次数
            sfi = max(1.0, min(sfi, 15.0))
        else:
            sfi = 4.5

        # ===== 心率变异性 RMSSD =====
        if heart_rate is not None and len(heart_rate) > 1:
            rr_intervals = 60000 / heart_rate  # bpm → ms
            diff = np.diff(rr_intervals)
            hrv_rmssd = np.sqrt(np.mean(diff ** 2))
            hrv_rmssd = max(10, min(hrv_rmssd, 100))
        else:
            hrv_rmssd = 48.0

        return {
            "sleep_efficiency": round(float(sleep_efficiency), 4),
            "deep_sleep_ratio": round(float(deep_sleep_ratio), 4),
            "sfi": round(float(sfi), 2),
            "hrv_rmssd": round(float(hrv_rmssd), 1),
        }

    def _generate_mock(self, date: str) -> dict:
        """生成模拟睡眠数据（开发/测试用）"""
        import numpy as np

        # 模拟夜间数据（同一个人基线 ± 随机波动）
        base_sleep_efficiency = 0.87
        base_deep_sleep = 0.30
        base_sfi = 4.5
        base_hrv = 48.0

        noise = np.random.RandomState(abs(hash(date)) % (2**31))

        return {
            "sleep_efficiency": round(
                max(0.60, min(0.98, base_sleep_efficiency + noise.normal(0, 0.04))), 4
            ),
            "deep_sleep_ratio": round(
                max(0.10, min(0.45, base_deep_sleep + noise.normal(0, 0.03))), 4
            ),
            "sfi": round(max(1.5, min(12.0, base_sfi + noise.normal(0, 1.0))), 2),
            "hrv_rmssd": round(max(20, min(80, base_hrv + noise.normal(0, 5))), 1),
        }

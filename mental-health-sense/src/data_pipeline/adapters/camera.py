"""
摄像头适配器（PIR 红外传感器 + IPC 骨骼追踪）

设备类型：
    - PIR 被动红外传感器（检测房间内是否有人）
    - IPC 网络摄像头 + 骨骼追踪算法（如 OpenPose / MediaPipe）

原始数据：
    - PIR：各房间的时间序列触发信号
    - IPC：人体骨骼关键点坐标序列

对接步骤：
    1. 确定摄像头型号和骨骼追踪方案
    2. 实现 _read_raw() 方法，连接摄像头数据流
    3. 在 _compute_features() 中实现活动量和空间转移熵的计算
    4. 将 mode 设为 "live"

Camera Adapter (PIR + IPC with skeletal tracking)

Device type:
    - PIR passive infrared sensors (room occupancy)
    - IPC camera with pose estimation (e.g. OpenPose / MediaPipe)

Raw data:
    - PIR: per-room time-series trigger signals
    - IPC: human skeletal keypoint coordinate sequences

Integration steps:
    1. Identify camera model and pose estimation pipeline
    2. Implement _read_raw() to connect to camera data stream
    3. Implement _compute_features() for activity level and spatial entropy
    4. Set mode to "live"
"""

from datetime import datetime

import numpy as np

from src.data_pipeline.adapters import SensorAdapter


class CameraAdapter(SensorAdapter):
    """
    PIR + IPC → 活动特征

    输入：PIR 触发记录 + IPC 骨骼关键点
    输出：{"daily_activity", "space_entropy"}

    Usage:
        # 开发/测试阶段
        adapter = CameraAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")

        # 对接真实摄像头后
        adapter = CameraAdapter(mode="live")
        features = adapter.extract(
            source={"pir": "/dev/pir0", "ipc": "rtsp://192.168.1.101/stream"},
            date="2026-08-01",
        )
    """

    FEATURE_NAMES = [
        "daily_activity",   # 日间活动量（步数/活动时长的综合指数）
        "space_entropy",    # 空间转移熵（房间穿梭的多样性）
    ]

    # 默认房间列表（根据实际部署配置）
    DEFAULT_ROOMS = ["bedroom", "living_room", "kitchen", "bathroom", "corridor"]

    def __init__(self, mode: str = "mock", rooms: list[str] | None = None):
        super().__init__(mode)
        self.rooms = rooms or self.DEFAULT_ROOMS

    def _read_raw(self, source: dict | str, date: str) -> dict:
        """
        【对接真实摄像头/PIR 时实现此方法】

        source 示例：
            {
                "pir": "mqtt://sensors/pir/#",        # PIR MQTT topic
                "ipc": "rtsp://192.168.1.101:554",   # IPC RTSP 流
            }

        原始数据（取决于设备，以下为典型格式）：
            raw = {
                "pir_events": [
                    {"room": "bedroom",   "time": "06:45:12", "duration_sec": 30},
                    {"room": "kitchen",   "time": "07:30:00", "duration_sec": 120},
                    ...
                ],
                "skeleton_frames": np.ndarray,  # (n_frames, n_joints, 3)  xyz坐标
                "activity_seconds": 18000,       # 总活动秒数
            }

        Returns:
            {"daily_activity": 5500, "space_entropy": 2.1}
        """
        # TODO: 对接真实摄像头/PIR
        # 方式1：从 MQTT 订阅 PIR 事件 + RTSP 拉流
        # 方式2：从本地文件读取录制的数据
        # 方式3：调用 IPC 厂商的 SDK
        raise NotImplementedError(
            "CameraAdapter._read_raw() — 请实现真实摄像头/PIR 对接逻辑。\n"
            "参考文档：src/data_pipeline/adapters/camera.py"
        )

    def _compute_features(self, raw_data: dict) -> dict:
        """
        从 PIR 事件和骨骼帧计算活动特征。

        Args:
            raw_data: SDK 返回的原始数据

        Returns:
            二维特征值 dict
        """
        pir_events = raw_data.get("pir_events", [])
        activity_seconds = raw_data.get("activity_seconds")
        skeleton_frames = raw_data.get("skeleton_frames")

        # ===== 日间活动量 =====
        if activity_seconds is not None:
            # 直接用 SDK 返回的总活动时长
            daily_activity = activity_seconds / 36  # 归一化到 0~10000 范围
        elif pir_events and len(pir_events) > 0:
            # 从 PIR 事件累计活动时长
            total_sec = sum(e.get("duration_sec", 0) for e in pir_events)
            daily_activity = total_sec / 36
        elif skeleton_frames is not None and len(skeleton_frames) > 0:
            # 从骨骼关键点位移估算活动量
            if len(skeleton_frames) > 1:
                displacement = np.sum(
                    np.sqrt(np.sum(np.diff(skeleton_frames, axis=0) ** 2, axis=-1))
                )
                daily_activity = displacement / 100  # 归一化
            else:
                daily_activity = 3000
        else:
            daily_activity = 5000  # 默认健康老人典型值

        daily_activity = max(100, min(20000, float(daily_activity)))

        # ===== 空间转移熵 =====
        if pir_events and len(pir_events) >= 2:
            room_sequence = [e.get("room", "unknown") for e in pir_events]

            # 统计各房间停留比例
            room_counts = {}
            for room in room_sequence:
                room_counts[room] = room_counts.get(room, 0) + 1
            total = sum(room_counts.values())

            # 香农熵
            entropy = 0.0
            for count in room_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)

            # 考虑房间切换次数
            transitions = sum(
                1 for i in range(1, len(room_sequence))
                if room_sequence[i] != room_sequence[i - 1]
            )
            space_entropy = entropy * (1 + transitions / max(len(room_sequence), 1))
            space_entropy = max(0.1, min(space_entropy, 5.0))
        else:
            space_entropy = 2.0

        return {
            "daily_activity": round(float(daily_activity), 1),
            "space_entropy": round(float(space_entropy), 2),
        }

    def _generate_mock(self, date: str) -> dict:
        """生成模拟活动数据"""
        import numpy as np

        noise = np.random.RandomState(abs(hash(f"activity_{date}")) % (2**31))

        return {
            "daily_activity": round(
                max(500, min(12000, 5500 + noise.normal(0, 800))), 1
            ),
            "space_entropy": round(
                max(0.5, min(4.5, 2.2 + noise.normal(0, 0.3))), 2
            ),
        }

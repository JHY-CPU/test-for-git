"""
拾音设备 + 智能音箱适配器

设备类型：
    - 拾音器（麦克风阵列，检测语音活动）
    - 智能音箱（带语音助手，记录交互日志）

原始数据：
    - 语音活动检测（VAD）片段序列
    - 智能音箱交互日志（对话轮次、时长）

对接步骤：
    1. 确定拾音设备和音箱型号
    2. 实现 _read_raw() 方法，从设备获取 VAD 片段和交互日志
    3. 在 _compute_features() 中计算对话轮次和说话时长占比
    4. 将 mode 设为 "live"

Microphone & Smart Speaker Adapter

Device type:
    - Microphone array (voice activity detection)
    - Smart speaker (interaction log)

Raw data:
    - VAD segment sequences (start_time, duration)
    - Smart speaker interaction logs (utterance count, duration)

Integration steps:
    1. Identify microphone and speaker models
    2. Implement _read_raw() to fetch VAD segments and interaction logs
    3. Implement _compute_features() for conversation turns and speech ratio
    4. Set mode to "live"
"""

import numpy as np

from src.data_pipeline.adapters import SensorAdapter


class MicrophoneAdapter(SensorAdapter):
    """
    拾音设备 + 智能音箱 → 社交特征

    输入：VAD 语音活动段 + 音箱交互日志
    输出：{"social_turns", "speech_duration_ratio"}

    Usage:
        # 开发/测试阶段
        adapter = MicrophoneAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")

        # 对接真实设备后
        adapter = MicrophoneAdapter(mode="live")
        features = adapter.extract(
            source={"mic": "/dev/audio0", "speaker": "192.168.1.102:8080"},
            date="2026-08-01",
        )
    """

    FEATURE_NAMES = [
        "social_turns",           # 对话轮次（每日交互次数）
        "speech_duration_ratio",  # 说话总时长 / 清醒时长
    ]

    # 假设清醒时长（小时），可根据实际睡眠数据调整
    DEFAULT_AWAKE_HOURS = 16

    def __init__(self, mode: str = "mock", awake_hours: float = 16):
        super().__init__(mode)
        self.awake_hours = awake_hours

    def _read_raw(self, source: dict | str, date: str) -> dict:
        """
        【对接真实拾音设备时实现此方法】

        source 示例：
            {
                "mic": "/dev/audio0",              # 拾音器设备
                "speaker": "http://192.168.1.102:8080/logs",  # 音箱日志 API
            }

        原始数据（取决于设备，以下为典型格式）：
            raw = {
                "vad_segments": [
                    {"start": "08:15:00", "duration_sec": 12.5, "speaker": "elder"},
                    {"start": "08:16:30", "duration_sec": 8.3,  "speaker": "elder"},
                    ...
                ],
                "speaker_interactions": [
                    {"time": "09:30", "type": "query", "utterances": 3},
                    {"time": "14:00", "type": "music", "utterances": 1},
                ],
            }

        Returns:
            {"social_turns": 30, "speech_duration_ratio": 0.15}
        """
        # TODO: 对接真实拾音设备
        # 方式1：持续录制 → VAD 处理 → 提取片段
        # 方式2：智能音箱 SDK 直接提供交互日志
        # 方式3：MQTT 订阅语音事件
        raise NotImplementedError(
            "MicrophoneAdapter._read_raw() — 请实现真实拾音设备对接逻辑。\n"
            "参考文档：src/data_pipeline/adapters/microphone.py"
        )

    def _compute_features(self, raw_data: dict) -> dict:
        """
        从 VAD 片段和音箱日志计算社交特征。

        Args:
            raw_data: SDK 返回的原始数据

        Returns:
            二维特征值 dict
        """
        vad_segments = raw_data.get("vad_segments", [])
        speaker_interactions = raw_data.get("speaker_interactions", [])

        # ===== 对话轮次 =====
        if vad_segments and len(vad_segments) > 0:
            # 从 VAD 的说话人切换中统计对话轮次
            # 简化：相邻片段间隔 > 3秒 视为新轮次
            social_turns = 1
            for i in range(1, len(vad_segments)):
                prev_end = vad_segments[i - 1].get("start", "00:00:00")
                curr_start = vad_segments[i].get("start", "00:00:00")

                # 有时间戳时精确计算间隔
                try:
                    from datetime import datetime
                    prev_dt = datetime.strptime(prev_end, "%H:%M:%S")
                    curr_dt = datetime.strptime(curr_start, "%H:%M:%S")
                    gap = (curr_dt - prev_dt).total_seconds()
                    if gap > 3:
                        social_turns += 1
                except (ValueError, TypeError):
                    social_turns += 1

            # 加上音箱交互
            for interaction in speaker_interactions:
                utterances = interaction.get("utterances", 0)
                social_turns += utterances
        elif speaker_interactions and len(speaker_interactions) > 0:
            social_turns = sum(
                si.get("utterances", 1) for si in speaker_interactions
            )
        else:
            social_turns = 25

        social_turns = max(0, min(200, int(social_turns)))

        # ===== 说话时长占比 =====
        if vad_segments and len(vad_segments) > 0:
            total_speech_sec = sum(
                seg.get("duration_sec", 0) for seg in vad_segments
            )
            awake_sec = self.awake_hours * 3600
            speech_duration_ratio = total_speech_sec / awake_sec if awake_sec > 0 else 0
            speech_duration_ratio = min(speech_duration_ratio, 0.5)
        else:
            speech_duration_ratio = 0.12  # 默认

        return {
            "social_turns": social_turns,
            "speech_duration_ratio": round(float(speech_duration_ratio), 4),
        }

    def _generate_mock(self, date: str) -> dict:
        """生成模拟社交数据"""
        import numpy as np

        noise = np.random.RandomState(abs(hash(f"social_{date}")) % (2**31))

        return {
            "social_turns": int(
                max(5, min(80, 30 + noise.normal(0, 6)))
            ),
            "speech_duration_ratio": round(
                max(0.01, min(0.35, 0.12 + noise.normal(0, 0.03))), 4
            ),
        }

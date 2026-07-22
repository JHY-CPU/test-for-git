"""
SenseVoice 声学/语义分析适配器

设备类型：麦克风阵列 + SenseVoice 语音情感分析模型
    SenseVoice 是阿里开源的语音情感识别模型，支持：
    - 情感分类（中性/开心/悲伤/愤怒/恐惧/厌恶/惊讶）
    - 语速检测
    - 基频（F0）提取
    - 非言语声音检测（哭声、笑声、叹气）

对接步骤：
    1. 部署 SenseVoice 模型（本地 GPU / 云端 API）
    2. 配置音频流输入（拾音器 → SenseVoice）
    3. 实现 _read_raw() 方法，调 SenseVoice 推理结果
    4. 将 mode 设为 "live"

SenseVoice Acoustic/Semantic Adapter

Device type: Microphone + SenseVoice speech emotion recognition model
    SenseVoice (Alibaba open source) supports:
    - Emotion classification (neutral/happy/sad/angry/fearful/disgusted/surprised)
    - Speech rate detection
    - Fundamental frequency (F0) extraction
    - Non-verbal sound detection (crying, laughing, sighing)

Integration steps:
    1. Deploy SenseVoice model (local GPU / cloud API)
    2. Configure audio stream input (microphone → SenseVoice)
    3. Implement _read_raw() to call SenseVoice inference results
    4. Set mode to "live"
"""

import numpy as np

from src.data_pipeline.adapters import SensorAdapter


class SenseVoiceAdapter(SensorAdapter):
    """
    拾音器 + SenseVoice → 声学/语义特征

    输入：音频流 → SenseVoice 推理结果
    输出：{"sad_ratio", "avg_speed", "pitch_variability", "distress_events"}

    Usage:
        # 开发/测试阶段
        adapter = SenseVoiceAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")

        # 对接 SenseVoice 后
        adapter = SenseVoiceAdapter(mode="live")
        features = adapter.extract(
            source="http://localhost:8765/sensevoice",  # SenseVoice API
            date="2026-08-01",
        )
    """

    FEATURE_NAMES = [
        "sad_ratio",          # 悲伤标签占比 [0, 1]
        "avg_speed",          # 平均语速（音节/秒）
        "pitch_variability",  # 基频变异性 F0标准差 (Hz)，语调单调性
        "distress_events",    # 叹气/哭声等非言语痛苦声音频次
    ]

    # SenseVoice 情感标签映射
    EMOTION_LABELS = {
        "neutral":   "neutral",
        "happy":     "happy",
        "sad":       "sad",
        "angry":     "angry",
        "fearful":   "fearful",
        "disgusted": "disgusted",
        "surprised": "surprised",
    }

    # 痛苦声学事件类型
    DISTRESS_SOUNDS = {"cry", "scream", "sigh", "groan", "moan"}

    def _read_raw(self, source: str, date: str) -> dict:
        """
        【对接 SenseVoice 时实现此方法】

        source 示例：
            - "http://localhost:8765/sensevoice"   # 本地 API
            - "grpc://sensevoice-server:50051"     # gRPC 服务
            - "/path/to/sensevoice_results/"      # 批量结果目录

        SenseVoice 推理结果（典型格式）：
            raw = {
                "utterances": [
                    {
                        "start": "08:15:00",
                        "duration_sec": 3.2,
                        "text": "今天天气不错",
                        "emotion": "neutral",        # 情感标签
                        "emotion_probs": {           # 各情感概率
                            "neutral": 0.82, "happy": 0.10, "sad": 0.03, ...
                        },
                        "speech_rate": 4.5,          # 音节/秒
                        "pitch_mean": 218.0,         # 平均F0 (Hz)
                        "non_verbal": None,          # 非言语声音类型
                    },
                    ...
                ],
                "non_verbal_events": [
                    {"start": "14:22:10", "type": "sigh", "duration_sec": 2.1},
                    ...
                ],
            }

        Returns:
            {
                "sad_ratio": 0.05,
                "avg_speed": 4.2,
                "pitch_variability": 28.0,
                "distress_events": 1,
            }
        """
        # TODO: 对接 SenseVoice
        # 方式1：部署 SenseVoice 本地服务 → HTTP API 调用
        # 方式2：gRPC 流式调用
        # 方式3：离线批量处理每日音频文件
        # 参考：https://github.com/FunAudioLLM/SenseVoice
        raise NotImplementedError(
            "SenseVoiceAdapter._read_raw() — 请实现 SenseVoice 对接逻辑。\n"
            "参考文档：src/data_pipeline/adapters/sensevoice.py\n"
            "SenseVoice GitHub: https://github.com/FunAudioLLM/SenseVoice"
        )

    def _compute_features(self, raw_data: dict) -> dict:
        """
        从 SenseVoice 推理结果计算声学特征。

        Args:
            raw_data: SenseVoice 推理结果

        Returns:
            四维特征值 dict
        """
        utterances = raw_data.get("utterances", [])
        non_verbal_events = raw_data.get("non_verbal_events", [])

        n_utterances = len(utterances)

        # ===== 悲伤占比 =====
        if n_utterances > 0:
            sad_count = 0
            sad_weight_sum = 0.0
            total_weight = 0.0

            for utt in utterances:
                emotion = utt.get("emotion", "neutral")
                probs = utt.get("emotion_probs", {})
                duration = utt.get("duration_sec", 1.0)

                if probs:
                    sad_prob = probs.get("sad", 0.0)
                    sad_weight_sum += sad_prob * duration
                elif emotion == "sad":
                    sad_count += 1

                total_weight += duration

            if sad_weight_sum > 0:
                sad_ratio = sad_weight_sum / total_weight
            else:
                sad_ratio = sad_count / n_utterances
        else:
            sad_ratio = 0.05

        sad_ratio = min(max(sad_ratio, 0.0), 1.0)

        # ===== 平均语速 =====
        if n_utterances > 0:
            speeds = [
                utt.get("speech_rate", 4.0) for utt in utterances
                if utt.get("speech_rate") is not None
            ]
            avg_speed = np.mean(speeds) if speeds else 4.0
        else:
            avg_speed = 4.0

        avg_speed = max(1.0, min(avg_speed, 8.0))

        # ===== 基频变异性（F0标准差）=====
        # 文献依据：抑郁表现为语调平淡单调，F0变异性下降（而非均值下降）
        # 参考 SD F0 与抑郁严重度相关（PubMed 38089742）
        if n_utterances >= 2:
            pitches = [
                utt.get("pitch_mean", 200.0) for utt in utterances
                if utt.get("pitch_mean") is not None
            ]
            pitch_variability = float(np.std(pitches)) if len(pitches) >= 2 else 25.0
        else:
            pitch_variability = 25.0

        pitch_variability = max(0.0, min(150.0, float(pitch_variability)))

        # ===== 痛苦事件频次 =====
        distress_events = len(non_verbal_events)
        # 也检查 utterances 中的 non_verbal 字段
        for utt in utterances:
            nv = utt.get("non_verbal")
            if nv and nv in self.DISTRESS_SOUNDS:
                distress_events += 1

        distress_events = max(0, min(distress_events, 100))

        return {
            "sad_ratio": round(float(sad_ratio), 4),
            "avg_speed": round(float(avg_speed), 2),
            "pitch_variability": round(float(pitch_variability), 1),
            "distress_events": int(distress_events),
        }

    def _generate_mock(self, date: str) -> dict:
        """生成模拟声学数据"""
        import numpy as np

        noise = np.random.RandomState(abs(hash(f"acoustic_{date}")) % (2**31))

        sad_ratio = max(0.0, min(0.3, 0.05 + noise.normal(0, 0.03)))
        # 有 5% 概率出现 distress 事件
        distress = 1 if noise.random() < 0.05 else 0

        return {
            "sad_ratio": round(float(sad_ratio), 4),
            "avg_speed": round(max(2.0, min(6.0, 4.2 + noise.normal(0, 0.4))), 2),
            "pitch_variability": round(max(5.0, min(60.0, 30 + noise.normal(0, 5))), 1),
            "distress_events": distress,
        }

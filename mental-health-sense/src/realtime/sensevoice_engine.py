"""
SenseVoice 实时推理引擎

基于 test_model.py 改造，支持：
    1. 流式音频输入
    2. 实时情感/声学特征提取
    3. 按时间窗口聚合特征
"""

import os
import re
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import numpy as np


class SenseVoiceEngine:
    """SenseVoice 实时推理引擎"""

    def __init__(
        self,
        model_cache_dir: Optional[str] = None,
        device: str = "cuda:0",
        batch_size: int = 15,
    ):
        """
        Args:
            model_cache_dir: 模型缓存目录
            device: 推理设备 ("cuda:0" 或 "cpu")
            batch_size: 批处理大小
        """
        self.device = device
        self.batch_size = batch_size

        # 配置模型缓存路径
        if model_cache_dir is None:
            model_cache_dir = os.path.join(os.path.dirname(__file__), "../../funasr_models")
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = model_cache_dir
        os.environ["HF_HOME"] = model_cache_dir

        # 抑制下载日志
        logging.getLogger("modelscope_hub.download").setLevel(logging.WARNING)

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载 SenseVoice 模型"""
        try:
            from funasr import AutoModel

            print("[SenseVoiceEngine] 正在加载模型...")
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
                vad_kwargs={"max_single_segment_time": 10000},
                device=self.device,
                disable_update=True,
            )
            print("[SenseVoiceEngine] 模型加载完成")
        except Exception as e:
            print(f"[SenseVoiceEngine] 模型加载失败: {e}")
            raise

    def process_audio(
        self,
        audio_input: str | np.ndarray,
        language: str = "zh",
    ) -> dict:
        """
        处理音频并提取特征

        Args:
            audio_input: 音频文件路径 或 numpy数组 (sample_rate=16000)
            language: 语言代码

        Returns:
            {
                "utterances": [
                    {
                        "start_sec": float,
                        "end_sec": float,
                        "duration_sec": float,
                        "text": str,
                        "emotion": str,
                        "speech_rate": float,  # 音节/秒
                        "pitch_mean": float,   # 基频F0
                    },
                    ...
                ],
                "non_verbal_events": [
                    {"start_sec": float, "type": str, "duration_sec": float},
                    ...
                ],
                "speaker_mapping": {spk_id: "speaker_1", ...}
            }
        """
        # 如果是numpy数组，需要保存为临时文件
        temp_file = None
        if isinstance(audio_input, np.ndarray):
            temp_file = self._save_temp_audio(audio_input)
            audio_input = temp_file

        try:
            # 调用 SenseVoice 推理
            res = self.model.generate(
                input=audio_input,
                language=language,
                use_itn=True,
                batch_size=self.batch_size,
                batch_size_type="sample",
            )

            # 解析结果
            result = self._parse_result(res)
            return result

        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def _save_temp_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """将numpy音频保存为临时wav文件"""
        from scipy.io import wavfile

        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        # 转换为int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(temp_path, sample_rate, audio_int16)

        return temp_path

    def _parse_result(self, res: list) -> dict:
        """解析 SenseVoice 原始输出"""
        result = {
            "utterances": [],
            "non_verbal_events": [],
            "speaker_mapping": {},
        }

        if not res or "sentence_info" not in res[0]:
            # 降级处理：无分段信息
            raw_text = res[0].get("text", "") if res else ""
            clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()
            if clean_text:
                result["utterances"].append({
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "text": clean_text,
                    "emotion": "neutral",
                    "speech_rate": 4.0,
                    "pitch_mean": 200.0,
                })
            return result

        # 处理分段信息
        speaker_role = {}
        for seg in res[0]["sentence_info"]:
            start_sec = seg.get('start', 0) / 1000.0
            end_sec = seg.get('end', 0) / 1000.0
            speaker_id = seg.get('spk', 0)
            raw_text = seg.get('sentence', '')

            # 清理文本中的情感标签
            clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()
            if not clean_text:
                continue

            # 说话人映射
            if speaker_id not in speaker_role:
                speaker_role[speaker_id] = f"speaker_{len(speaker_role) + 1}"

            # 提取情感标签（从原始文本中）
            emotion = self._extract_emotion_from_text(raw_text)

            # 估算语速（简单估算：字数/时长）
            duration = max(0.1, end_sec - start_sec)
            speech_rate = len(clean_text) / duration

            # 基频占位（SenseVoice Small 不直接提供F0，需要额外提取）
            pitch_mean = 200.0  # 默认值

            result["utterances"].append({
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(duration, 3),
                "text": clean_text,
                "speaker": speaker_role[speaker_id],
                "emotion": emotion,
                "speech_rate": round(speech_rate, 2),
                "pitch_mean": pitch_mean,
            })

        result["speaker_mapping"] = speaker_role
        return result

    def _extract_emotion_from_text(self, text: str) -> str:
        """从SenseVoice输出的标签文本中提取情感"""
        emotion_tags = {
            "<|HAPPY|>": "happy",
            "<|SAD|>": "sad",
            "<|ANGRY|>": "angry",
            "<|NEUTRAL|>": "neutral",
            "<|FEARFUL|>": "fearful",
            "<|DISGUSTED|>": "disgusted",
            "<|SURPRISED|>": "surprised",
        }

        for tag, emotion in emotion_tags.items():
            if tag in text:
                return emotion

        return "neutral"

    def compute_acoustic_features(self, result: dict) -> dict:
        """
        从推理结果计算声学特征（用于心理健康分析）

        Returns:
            {
                "sad_ratio": float,          # 悲伤占比 [0, 1]
                "avg_speed": float,          # 平均语速 (字/秒)
                "pitch_variability": float,  # 基频变异性 F0标准差 (Hz)
                "distress_events": int,      # 痛苦事件次数
            }
        """
        utterances = result.get("utterances", [])
        n_utterances = len(utterances)

        if n_utterances == 0:
            return {
                "sad_ratio": 0.05,
                "avg_speed": 4.0,
                "pitch_variability": 25.0,
                "distress_events": 0,
            }

        # 1. 悲伤占比（加权计算）
        sad_weight = 0.0
        total_weight = 0.0
        for utt in utterances:
            duration = utt.get("duration_sec", 1.0)
            emotion = utt.get("emotion", "neutral")
            if emotion == "sad":
                sad_weight += duration
            total_weight += duration

        sad_ratio = sad_weight / total_weight if total_weight > 0 else 0.0
        sad_ratio = min(max(sad_ratio, 0.0), 1.0)

        # 2. 平均语速
        speeds = [utt.get("speech_rate", 4.0) for utt in utterances]
        avg_speed = np.mean(speeds) if speeds else 4.0
        avg_speed = max(1.0, min(8.0, float(avg_speed)))

        # 3. 基频变异性（F0标准差，语调单调性）
        pitches = [utt.get("pitch_mean", 200.0) for utt in utterances]
        pitch_variability = float(np.std(pitches)) if len(pitches) >= 2 else 25.0
        pitch_variability = max(0.0, min(150.0, pitch_variability))

        # 4. 痛苦事件（简单检测：悲伤/愤怒/恐惧情绪）
        distress_emotions = {"sad", "angry", "fearful"}
        distress_events = sum(
            1 for utt in utterances if utt.get("emotion") in distress_emotions
        )

        return {
            "sad_ratio": round(sad_ratio, 4),
            "avg_speed": round(avg_speed, 2),
            "pitch_variability": round(pitch_variability, 1),
            "distress_events": distress_events,
        }


class RealtimeFeatureAggregator:
    """实时特征聚合器 - 按时间窗口累积特征"""

    def __init__(self, window_hours: int = 24):
        """
        Args:
            window_hours: 时间窗口（小时），默认24小时=1天
        """
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600

        # 存储最近的utterances
        self.utterances_buffer = []  # [(timestamp, utterance_dict), ...]

    def add_utterances(self, utterances: list[dict], timestamp: float):
        """
        添加新的语音片段

        Args:
            utterances: SenseVoice输出的utterances列表
            timestamp: 音频开始时间戳（秒）
        """
        for utt in utterances:
            utt_timestamp = timestamp + utt.get("start_sec", 0)
            self.utterances_buffer.append((utt_timestamp, utt))

        # 清理过期数据
        self._cleanup_old_data(timestamp)

    def _cleanup_old_data(self, current_time: float):
        """移除超出时间窗口的数据"""
        cutoff_time = current_time - self.window_seconds
        self.utterances_buffer = [
            (ts, utt) for ts, utt in self.utterances_buffer if ts >= cutoff_time
        ]

    def get_current_features(self) -> dict:
        """
        获取当前时间窗口的声学特征

        Returns:
            {
                "sad_ratio": float,
                "avg_speed": float,
                "pitch_variability": float,
                "distress_events": int,
                "n_utterances": int,
                "total_duration": float,
            }
        """
        if not self.utterances_buffer:
            return {
                "sad_ratio": 0.05,
                "avg_speed": 4.0,
                "pitch_variability": 25.0,
                "distress_events": 0,
                "n_utterances": 0,
                "total_duration": 0.0,
            }

        utterances = [utt for _, utt in self.utterances_buffer]
        n_utterances = len(utterances)

        # 1. 悲伤占比
        sad_weight = 0.0
        total_weight = 0.0
        for utt in utterances:
            duration = utt.get("duration_sec", 1.0)
            emotion = utt.get("emotion", "neutral")
            if emotion == "sad":
                sad_weight += duration
            total_weight += duration

        sad_ratio = sad_weight / total_weight if total_weight > 0 else 0.0

        # 2. 平均语速
        speeds = [utt.get("speech_rate", 4.0) for utt in utterances]
        avg_speed = float(np.mean(speeds)) if speeds else 4.0

        # 3. 基频变异性（F0标准差）
        pitches = [utt.get("pitch_mean", 200.0) for utt in utterances]
        pitch_variability = float(np.std(pitches)) if len(pitches) >= 2 else 25.0

        # 4. 痛苦事件
        distress_emotions = {"sad", "angry", "fearful"}
        distress_events = sum(
            1 for utt in utterances if utt.get("emotion") in distress_emotions
        )

        return {
            "sad_ratio": round(sad_ratio, 4),
            "avg_speed": round(avg_speed, 2),
            "pitch_variability": round(pitch_variability, 1),
            "distress_events": distress_events,
            "n_utterances": n_utterances,
            "total_duration": round(total_weight, 1),
        }

    def get_hourly_features(self, last_n_hours: int = 1) -> dict:
        """获取最近N小时的特征（用于短期趋势分析）"""
        if not self.utterances_buffer:
            return self.get_current_features()

        # 过滤最近N小时的数据
        current_time = self.utterances_buffer[-1][0] if self.utterances_buffer else 0
        cutoff_time = current_time - (last_n_hours * 3600)

        recent_utterances = [
            utt for ts, utt in self.utterances_buffer if ts >= cutoff_time
        ]

        if not recent_utterances:
            return self.get_current_features()

        # 计算特征（与get_current_features相同逻辑）
        n_utterances = len(recent_utterances)

        sad_weight = sum(
            utt.get("duration_sec", 1.0)
            for utt in recent_utterances
            if utt.get("emotion") == "sad"
        )
        total_weight = sum(utt.get("duration_sec", 1.0) for utt in recent_utterances)
        sad_ratio = sad_weight / total_weight if total_weight > 0 else 0.0

        speeds = [utt.get("speech_rate", 4.0) for utt in recent_utterances]
        avg_speed = float(np.mean(speeds)) if speeds else 4.0

        pitches = [utt.get("pitch_mean", 200.0) for utt in recent_utterances]
        pitch_variability = float(np.std(pitches)) if len(pitches) >= 2 else 25.0

        distress_emotions = {"sad", "angry", "fearful"}
        distress_events = sum(
            1 for utt in recent_utterances if utt.get("emotion") in distress_emotions
        )

        return {
            "sad_ratio": round(sad_ratio, 4),
            "avg_speed": round(avg_speed, 2),
            "pitch_variability": round(pitch_variability, 1),
            "distress_events": distress_events,
            "n_utterances": n_utterances,
            "total_duration": round(total_weight, 1),
        }


# 使用示例
if __name__ == "__main__":
    # 示例1：处理音频文件
    print("=== 测试 SenseVoice 引擎 ===")
    engine = SenseVoiceEngine(device="cuda:0")

    # 处理音频
    test_audio = r"C:\Users\21308\OneDrive\Desktop\TEST\TEST\tts_test1702\tts_test1702.mp3"
    if os.path.exists(test_audio):
        result = engine.process_audio(test_audio)
        print(f"\n检测到 {len(result['utterances'])} 段语音")

        # 计算特征
        features = engine.compute_acoustic_features(result)
        print("\n声学特征:")
        print(json.dumps(features, ensure_ascii=False, indent=2))

    # 示例2：实时特征聚合
    print("\n=== 测试实时聚合器 ===")
    aggregator = RealtimeFeatureAggregator(window_hours=24)

    # 模拟添加数据
    mock_utterances = [
        {"start_sec": 0, "duration_sec": 3.2, "emotion": "neutral", "speech_rate": 4.5, "pitch_mean": 215},
        {"start_sec": 3.2, "duration_sec": 2.8, "emotion": "sad", "speech_rate": 3.8, "pitch_mean": 195},
    ]
    aggregator.add_utterances(mock_utterances, timestamp=time.time())

    current_features = aggregator.get_current_features()
    print("\n当前24小时特征:")
    print(json.dumps(current_features, ensure_ascii=False, indent=2))

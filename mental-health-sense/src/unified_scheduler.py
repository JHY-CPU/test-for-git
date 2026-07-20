"""
统一调度器 - 整合实时监测与每日推理

将实时监测和每日推理整合为一个统一的系统
"""

import time
import threading
from datetime import datetime
from pathlib import Path

from src.unified_data_manager import UnifiedDataManager
from src.realtime.audio_stream import AudioStream
from src.realtime.sensevoice_engine import SenseVoiceEngine


class UnifiedScheduler:
    """统一调度器

    职责：
        1. 管理实时音频采集和推理
        2. 定时触发每日推理（使用实时累积的数据）
        3. 协调实时预警和每日预警
    """

    def __init__(
        self,
        elder_id: str,
        audio_stream: AudioStream,
        daily_inference_time: str = "02:00",  # 每日推理时间
    ):
        self.elder_id = elder_id
        self.audio_stream = audio_stream
        self.daily_inference_time = daily_inference_time

        # 统一数据管理器
        self.data_manager = UnifiedDataManager(elder_id)

        # SenseVoice推理引擎
        self.sensevoice = SenseVoiceEngine(device="cuda:0")

        # 控制标志
        self._stop_event = threading.Event()
        self._threads = []

    def start(self):
        """启动统一系统"""
        print(f"[UnifiedScheduler] 启动老人 {self.elder_id} 的监测系统")

        # 启动音频流
        self.audio_stream.start()

        # 启动实时处理线程
        realtime_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        realtime_thread.start()
        self._threads.append(realtime_thread)

        # 启动实时风险检查线程
        risk_thread = threading.Thread(target=self._risk_check_loop, daemon=True)
        risk_thread.start()
        self._threads.append(risk_thread)

        # 启动每日调度线程
        daily_thread = threading.Thread(target=self._daily_schedule_loop, daemon=True)
        daily_thread.start()
        self._threads.append(daily_thread)

        print("[UnifiedScheduler] 系统已启动")

    def stop(self):
        """停止系统"""
        print("[UnifiedScheduler] 停止系统...")
        self._stop_event.set()
        self.audio_stream.stop()

        for t in self._threads:
            t.join(timeout=5)

        print("[UnifiedScheduler] 系统已停止")

    def _realtime_loop(self):
        """实时音频处理循环"""
        print("[实时轨] 开始处理音频流...")

        while not self._stop_event.is_set():
            try:
                # 获取音频数据
                audio_data = self.audio_stream.get_audio(timeout=1.0)
                if audio_data is None:
                    continue

                audio_chunk, timestamp = audio_data

                # SenseVoice推理
                result = self.sensevoice.process_audio(audio_chunk)
                utterances = result.get("utterances", [])

                if utterances:
                    # 添加到统一数据管理器
                    self.data_manager.add_realtime_utterances(utterances, timestamp)

                    print(f"[实时轨] 处理 {len(utterances)} 段语音，"
                          f"时间戳 {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")

            except Exception as e:
                print(f"[实时轨] 处理异常: {e}")
                time.sleep(1)

    def _risk_check_loop(self):
        """实时风险检查循环（每小时）"""
        print("[实时轨] 启动风险检查...")

        while not self._stop_event.is_set():
            try:
                # 等待1小时
                time.sleep(3600)

                # 获取当前24小时特征
                features = self.data_manager.get_current_features()

                # 简化风险判定
                risk_level = self._quick_risk_assessment(features)

                if risk_level > 0:
                    self._trigger_realtime_alert(risk_level, features)

            except Exception as e:
                print(f"[实时轨] 风险检查异常: {e}")

    def _daily_schedule_loop(self):
        """每日推理调度循环（每天凌晨2点）"""
        print(f"[每日轨] 启动每日调度（执行时间: {self.daily_inference_time}）")

        last_run_date = None

        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                current_date = now.strftime("%Y-%m-%d")

                # 检查是否到达执行时间
                if current_time == self.daily_inference_time and current_date != last_run_date:
                    print(f"[每日轨] 开始每日推理 ({current_date})")

                    # 执行每日推理（使用实时累积的数据）
                    self._run_daily_inference(current_date)

                    last_run_date = current_date

                # 每分钟检查一次
                time.sleep(60)

            except Exception as e:
                print(f"[每日轨] 调度异常: {e}")

    def _run_daily_inference(self, date: str):
        """执行每日推理（整合实时数据）"""
        print(f"[每日轨] 执行每日推理: {date}")

        # 1. 从统一数据管理器获取声学特征
        acoustic_data = self.data_manager.get_daily_acoustic_data(date)
        print(f"[每日轨] 声学特征（来自实时系统）:")
        print(f"  - sad_ratio: {acoustic_data['sad_ratio']:.3f}")
        print(f"  - avg_speed: {acoustic_data['avg_speed']:.2f}")
        print(f"  - avg_pitch: {acoustic_data['avg_pitch']:.1f}")
        print(f"  - distress_events: {acoustic_data['distress_events']}")

        # 2. 获取其他传感器数据（TODO: 对接其他传感器）
        sleep_data = self._get_sleep_data(date)
        activity_data = self._get_activity_data(date)
        social_data = self._get_social_data(date)

        # 3. 聚合为10维特征向量
        from src.data_pipeline.aggregator import aggregate_daily_features

        try:
            daily_vector = aggregate_daily_features(
                date_str=date,
                acoustic_data=acoustic_data,  # 来自实时系统！
                sleep_data=sleep_data,
                activity_data=activity_data,
                social_data=social_data,
            )

            print(f"[每日轨] 10维特征向量已生成")

            # 4. GRU推理（TODO: 调用现有的GRU模型）
            # from src.baseline.inference import daily_inference
            # result = daily_inference(self.elder_id, date, daily_vector)

            print(f"[每日轨] GRU推理完成（待实现）")

        except Exception as e:
            print(f"[每日轨] 推理失败: {e}")

    def _quick_risk_assessment(self, features: dict) -> int:
        """快速风险评估（实时系统用）"""
        sad_ratio = features.get("sad_ratio", 0)
        avg_speed = features.get("avg_speed", 4.0)
        distress_events = features.get("distress_events", 0)

        # 简化判定
        if sad_ratio > 0.30 and avg_speed < 3.5 and distress_events > 5:
            return 3  # 严重
        elif sad_ratio > 0.25 and avg_speed < 3.8:
            return 2  # 提醒
        elif sad_ratio > 0.20 or distress_events > 3:
            return 1  # 关注
        else:
            return 0  # 正常

    def _trigger_realtime_alert(self, risk_level: int, features: dict):
        """触发实时预警"""
        risk_names = ["正常", "关注", "提醒", "严重"]
        print(f"\n[实时预警] 风险等级: {risk_names[risk_level]} (Level {risk_level})")
        print(f"  - 悲伤占比: {features['sad_ratio']:.3f}")
        print(f"  - 平均语速: {features['avg_speed']:.2f}")
        print(f"  - 痛苦事件: {features['distress_events']}")

        # TODO: 推送到App/短信

    def _get_sleep_data(self, date: str) -> dict:
        """获取睡眠数据（TODO: 对接睡眠雷达）"""
        # 返回模拟数据
        return {
            "sleep_efficiency": 0.85,
            "deep_sleep_ratio": 0.25,
            "sfi": 15,
            "hrv_rmssd": 45,
        }

    def _get_activity_data(self, date: str) -> dict:
        """获取活动数据（TODO: 对接PIR+IPC）"""
        return {
            "daily_activity": 5000,
        }

    def _get_social_data(self, date: str) -> dict:
        """获取社交数据（TODO: 对接拾音+音箱）"""
        return {
            "social_turns": 15,
        }


# 使用示例
if __name__ == "__main__":
    from src.realtime.audio_stream import FileSimulatorStream

    # 创建音频流（使用文件模拟）
    audio_stream = FileSimulatorStream(
        audio_file="test_audio.mp3",
        loop=True,
        chunk_duration=10
    )

    # 创建统一调度器
    scheduler = UnifiedScheduler(
        elder_id="E001",
        audio_stream=audio_stream,
    )

    try:
        # 启动系统
        scheduler.start()

        # 运行一段时间
        print("\n系统运行中... (按 Ctrl+C 停止)\n")
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        scheduler.stop()

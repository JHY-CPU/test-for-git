"""
统一调度器 - 整合实时采集与每日趋势推理

将实时语音采集与每日趋势推理整合为一个系统：
    - 实时轨仅作**语音采集前端**，持续抽取声学特征供每日轨使用
    - 每日轨（GRU + EWMA + 连续偏离判定）在累积数据上做趋势预警

设计说明：实时轨**不做主动报警**。所有心理风险预警统一由每日趋势轨发出，
避免实时轨因单点波动过度敏感、频繁打扰家人。
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
        1. 管理实时音频采集和推理（仅采集语音特征，不做实时报警）
        2. 定时触发每日趋势推理（使用实时累积的数据）
        3. 预警统一由每日趋势轨发出
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

        # 启动实时采集线程（仅采集语音特征，不做实时报警）
        realtime_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        realtime_thread.start()
        self._threads.append(realtime_thread)

        # 启动每日调度线程（预警统一由每日趋势轨发出）
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
        """执行每日推理：真正调用 run_daily_pipeline（GRU + EWMA + 风险判定 + 预警）。

        声学特征来自实时轨的自然日聚合；睡眠/活动/社交从 data/raw/ 读取（真实
        传感器数据放到该目录即可）。任一路缺失时由 run_daily_pipeline 内部据实
        判定数据质量（不再用假 mock 常量掩盖缺失）。
        """
        print(f"[每日轨] 执行每日推理: {date}")

        # 1. 声学特征：来自实时轨自然日聚合（带质量标记）
        acoustic = self.data_manager.get_daily_acoustic_with_quality(date)
        if acoustic.get("data_quality") == "missing":
            print(f"[每日轨] ⚠️ {date} 无实时声学数据，声学置空交由数据校验降级处理")
            acoustic_data = None
        else:
            acoustic_data = {k: acoustic[k] for k in
                             ("sad_ratio", "avg_speed", "pitch_variability", "distress_events")}
            print(f"[每日轨] 声学特征（来自实时轨）: sad={acoustic_data['sad_ratio']:.3f}, "
                  f"speed={acoustic_data['avg_speed']:.2f}, "
                  f"pitchVar={acoustic_data['pitch_variability']:.1f}, "
                  f"distress={acoustic_data['distress_events']}")

        # 2. 其他三路传感器：从 data/raw/ 读取真实数据（无则为 None）
        from src.scheduler.daily_job import load_raw_sensors, run_daily_pipeline
        raw = load_raw_sensors(self.elder_id, date)
        raw_data = {
            "acoustic": acoustic_data,
            "sleep": raw["sleep"],
            "activity": raw["activity"],
            "social": raw["social"],
        }

        # 3. 调用完整每日管道（聚合→填充→校验→保存→GRU推理→风险判定→预警）
        try:
            result = run_daily_pipeline(
                elder_id=self.elder_id,
                date_str=date,
                raw_data=raw_data,
            )
            risk = result.get("risk_result") or {}
            print(f"[每日轨] 完成: status={result.get('status')}, "
                  f"quality={result.get('data_quality')}, "
                  f"risk_level={risk.get('risk_level', '-')}")
        except Exception as e:
            print(f"[每日轨] 推理失败: {e}")


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

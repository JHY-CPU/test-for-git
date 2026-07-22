"""
实时语音采集前端 - 主控制器

职责：作为每日趋势轨的语音**采集前端**，持续听取语音、用 SenseVoice 抽取
声学特征、聚合成 24 小时快照供每日轨读取。

工作流程：
    1. 音频流采集（麦克风/摄像头）
    2. SenseVoice实时推理
    3. 特征累积（24小时窗口）
    4. 周期性保存特征快照（供每日趋势轨读取）

设计说明：本模块**不做实时主动报警**。所有心理风险预警统一由每日趋势轨
（GRU + EWMA + 连续偏离判定）在累积数据上判定后发出，避免实时轨因单点
波动过度敏感、频繁打扰家人。急性求救类信号如需处理，应作为独立能力另行设计。
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from src.realtime.audio_stream import AudioStream, MicrophoneStream, FileSimulatorStream
from src.realtime.sensevoice_engine import SenseVoiceEngine, RealtimeFeatureAggregator


class RealtimeMonitor:
    """实时心理健康监测控制器"""

    def __init__(
        self,
        elder_id: str,
        audio_stream: AudioStream,
        output_dir: str = "./data/realtime",
        save_interval: int = 1800,  # 每30分钟保存一次特征
    ):
        """
        Args:
            elder_id: 老人ID
            audio_stream: 音频流对象
            output_dir: 输出目录
            save_interval: 特征保存间隔（秒）
        """
        self.elder_id = elder_id
        self.audio_stream = audio_stream
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = self.output_dir / elder_id / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        print(f"[RealtimeMonitor] 初始化老人 {elder_id} 的语音采集前端...")
        self.engine = SenseVoiceEngine(device="cuda:0")
        self.aggregator = RealtimeFeatureAggregator(window_hours=24)

        # 控制标志
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._save_thread: Optional[threading.Thread] = None

        # 统计信息
        self.stats = {
            "start_time": None,
            "audio_chunks_processed": 0,
            "total_utterances": 0,
            "total_duration": 0.0,
        }

    def start(self):
        """启动实时监测"""
        print(f"[RealtimeMonitor] 启动监测系统 (elder_id={self.elder_id})")
        self.stats["start_time"] = datetime.now()

        # 启动音频流
        self.audio_stream.start()

        # 启动处理线程
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()

        # 启动定期保存线程（保存特征快照供每日趋势轨读取）
        self._save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self._save_thread.start()

        print("[RealtimeMonitor] 采集前端已启动，开始采集语音特征...")

    def stop(self):
        """停止监测"""
        print("[RealtimeMonitor] 正在停止监测...")
        self._stop_event.set()

        # 停止音频流
        self.audio_stream.stop()

        # 等待线程结束
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        if self._save_thread:
            self._save_thread.join(timeout=5)

        # 最后保存一次
        self._save_current_features()

        print("[RealtimeMonitor] 已停止")

    def _process_loop(self):
        """音频处理主循环"""
        while not self._stop_event.is_set():
            try:
                # 从音频流获取数据
                audio_data = self.audio_stream.get_audio(timeout=1.0)
                if audio_data is None:
                    continue

                audio_chunk, timestamp = audio_data

                # SenseVoice 推理
                result = self.engine.process_audio(audio_chunk)

                # 添加到聚合器
                utterances = result.get("utterances", [])
                if utterances:
                    self.aggregator.add_utterances(utterances, timestamp)

                    # 更新统计
                    self.stats["audio_chunks_processed"] += 1
                    self.stats["total_utterances"] += len(utterances)
                    self.stats["total_duration"] += sum(
                        u.get("duration_sec", 0) for u in utterances
                    )

                    # 打印日志
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"处理音频块 #{self.stats['audio_chunks_processed']}: "
                        f"{len(utterances)} 段语音"
                    )

            except Exception as e:
                print(f"[RealtimeMonitor] 处理异常: {e}")
                time.sleep(1)

    def _save_loop(self):
        """定期保存循环"""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.save_interval)
                self._save_current_features()
            except Exception as e:
                print(f"[RealtimeMonitor] 保存异常: {e}")

    def _save_current_features(self):
        """保存当前特征到文件"""
        features = self.aggregator.get_current_features()
        timestamp = datetime.now()

        # 保存为JSON（按日期分文件）
        date_str = timestamp.strftime("%Y-%m-%d")
        feature_file = self.features_dir / f"features_{date_str}.json"

        # 追加模式
        if feature_file.exists():
            with open(feature_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"date": date_str, "elder_id": self.elder_id, "records": []}

        data["records"].append({
            "timestamp": timestamp.isoformat(),
            **features,
        })

        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats["start_time"]:
            runtime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["runtime_seconds"] = runtime
            stats["runtime_str"] = str(timedelta(seconds=int(runtime)))
        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"监测统计 - {self.elder_id}")
        print(f"{'='*60}")
        print(f"运行时长: {stats.get('runtime_str', 'N/A')}")
        print(f"音频块处理: {stats['audio_chunks_processed']} 个")
        print(f"语音片段: {stats['total_utterances']} 段")
        print(f"总时长: {stats['total_duration']:.1f} 秒")
        print(f"{'='*60}\n")


# 使用示例
if __name__ == "__main__":
    import sys

    print("实时心理健康监测系统")
    print("=" * 60)

    # 选择音频源
    print("\n选择音频输入源:")
    print("1. 麦克风实时采集")
    print("2. 文件模拟流（测试）")

    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        # 麦克风模式
        audio_stream = MicrophoneStream(chunk_duration=10)
    else:
        # 文件模拟模式
        test_file = r"C:\Users\21308\OneDrive\Desktop\TEST\TEST\tts_test1702\tts_test1702.mp3"
        if not Path(test_file).exists():
            print(f"测试文件不存在: {test_file}")
            print("请修改 test_file 路径或选择麦克风模式")
            sys.exit(1)
        audio_stream = FileSimulatorStream(test_file, loop=True, chunk_duration=10)

    # 创建采集前端
    monitor = RealtimeMonitor(
        elder_id="E001",
        audio_stream=audio_stream,
        save_interval=180,  # 3分钟保存一次
    )

    try:
        # 启动监测
        monitor.start()

        # 运行一段时间
        print("\n监测运行中... (按 Ctrl+C 停止)\n")
        while True:
            time.sleep(30)
            monitor.print_stats()

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        monitor.stop()
        monitor.print_stats()
        print("监测已结束")

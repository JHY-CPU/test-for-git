#!/usr/bin/env python3
"""
统一监测系统 - 启动脚本

整合实时监测与每日推理，形成一体化系统
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unified_scheduler import UnifiedScheduler
from src.realtime import MicrophoneStream, RTSPStream, FileSimulatorStream


def main():
    parser = argparse.ArgumentParser(description="统一心理健康监测系统")

    # 基本参数
    parser.add_argument(
        "--elder-id",
        type=str,
        default="E001",
        help="老人ID",
    )

    # 音频输入源
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument("--microphone", action="store_true", help="麦克风采集")
    audio_group.add_argument("--rtsp", type=str, help="RTSP摄像头地址")
    audio_group.add_argument("--file", type=str, help="音频文件路径（测试用）")

    # 调度参数
    parser.add_argument(
        "--daily-time",
        type=str,
        default="02:00",
        help="每日推理时间（HH:MM），默认: 02:00",
    )

    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=10,
        help="音频分段时长（秒），默认: 10",
    )

    args = parser.parse_args()

    # 打印启动信息
    print("=" * 70)
    print("统一心理健康监测系统")
    print("=" * 70)
    print(f"\n老人ID: {args.elder_id}")
    print(f"每日推理时间: {args.daily_time}")

    # 创建音频流
    if args.microphone:
        print("音频源: 麦克风")
        audio_stream = MicrophoneStream(chunk_duration=args.chunk_duration)
    elif args.rtsp:
        print(f"音频源: RTSP ({args.rtsp})")
        audio_stream = RTSPStream(rtsp_url=args.rtsp, chunk_duration=args.chunk_duration)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"\n错误: 文件不存在 - {args.file}")
            sys.exit(1)
        print(f"音频源: 文件模拟 ({args.file})")
        audio_stream = FileSimulatorStream(
            audio_file=args.file,
            loop=True,
            chunk_duration=args.chunk_duration
        )
    else:
        print("\n错误: 必须指定音频输入源")
        parser.print_help()
        sys.exit(1)

    print("\n系统架构:")
    print("  [实时轨] 持续采集音频 -> SenseVoice推理 -> 24h累积")
    print("           ├─ 每小时风险检查（快速响应）")
    print("           └─ 数据自动保存")
    print("  [每日轨] 每天02:00读取实时数据 -> 10维向量 -> GRU推理")
    print("           └─ 深度分析 + EWMA判定\n")

    # 创建统一调度器
    try:
        scheduler = UnifiedScheduler(
            elder_id=args.elder_id,
            audio_stream=audio_stream,
            daily_inference_time=args.daily_time,
        )
    except Exception as e:
        print(f"\n错误: 初始化失败 - {e}")
        sys.exit(1)

    # 启动系统
    try:
        scheduler.start()
        print("✓ 系统已启动\n")
        print("按 Ctrl+C 停止\n")
        print("-" * 70)

        # 主循环
        import time
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n收到停止信号...")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        scheduler.stop()
        print("\n系统已停止")


if __name__ == "__main__":
    main()

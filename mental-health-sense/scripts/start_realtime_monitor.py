#!/usr/bin/env python3
"""
实时心理健康监测系统 - 启动脚本

快速启动实时监测服务，支持多种配置模式
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime import (
    RealtimeMonitor,
    MicrophoneStream,
    RTSPStream,
    FileSimulatorStream,
)


def main():
    parser = argparse.ArgumentParser(description="实时心理健康监测系统")

    # 基本参数
    parser.add_argument(
        "--elder-id",
        type=str,
        default="E001",
        help="老人ID（默认: E001）",
    )

    # 音频输入源
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--microphone",
        action="store_true",
        help="使用麦克风采集",
    )
    audio_group.add_argument(
        "--rtsp",
        type=str,
        metavar="URL",
        help="RTSP摄像头地址，例如 rtsp://192.168.1.100:554/stream",
    )
    audio_group.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="音频文件路径（模拟实时流，用于测试）",
    )

    # 高级参数
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="麦克风设备索引（默认: None=系统默认麦克风）",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=10,
        help="音频分段时长（秒），默认: 10",
    )
    parser.add_argument(
        "--risk-check-interval",
        type=int,
        default=3600,
        help="风险检查间隔（秒），默认: 3600（1小时）",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1800,
        help="特征保存间隔（秒），默认: 1800（30分钟）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/realtime",
        help="输出目录，默认: ./data/realtime",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="GPU设备（cuda:0 或 cpu），默认: cuda:0",
    )

    args = parser.parse_args()

    # 创建音频流
    print("=" * 70)
    print("实时心理健康监测系统")
    print("=" * 70)
    print(f"\n老人ID: {args.elder_id}")

    if args.microphone:
        print(f"音频源: 麦克风 (device_index={args.device_index})")
        audio_stream = MicrophoneStream(
            device_index=args.device_index,
            chunk_duration=args.chunk_duration,
        )
    elif args.rtsp:
        print(f"音频源: RTSP摄像头 ({args.rtsp})")
        audio_stream = RTSPStream(
            rtsp_url=args.rtsp,
            chunk_duration=args.chunk_duration,
        )
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"\n❌ 错误: 音频文件不存在: {args.file}")
            sys.exit(1)
        print(f"音频源: 文件模拟 ({args.file})")
        audio_stream = FileSimulatorStream(
            audio_file=args.file,
            loop=True,
            chunk_duration=args.chunk_duration,
        )
    else:
        print("\n❌ 错误: 必须指定音频输入源")
        parser.print_help()
        sys.exit(1)

    print(f"音频分段: {args.chunk_duration} 秒")
    print(f"风险检查: 每 {args.risk_check_interval} 秒")
    print(f"特征保存: 每 {args.save_interval} 秒")
    print(f"输出目录: {args.output_dir}")
    print(f"GPU设备: {args.gpu}")
    print()

    # 创建监测器
    try:
        monitor = RealtimeMonitor(
            elder_id=args.elder_id,
            audio_stream=audio_stream,
            output_dir=args.output_dir,
            risk_check_interval=args.risk_check_interval,
            save_interval=args.save_interval,
        )
    except Exception as e:
        print(f"\n❌ 监测器初始化失败: {e}")
        sys.exit(1)

    # 启动监测
    try:
        monitor.start()
        print("✅ 监测系统已启动")
        print("\n按 Ctrl+C 停止监测\n")
        print("-" * 70)

        # 主循环
        import time
        while True:
            time.sleep(60)  # 每分钟输出一次统计
            monitor.print_stats()

    except KeyboardInterrupt:
        print("\n\n⚠️  收到停止信号，正在关闭...")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
    finally:
        monitor.stop()
        monitor.print_stats()
        print("\n✅ 监测系统已停止")


if __name__ == "__main__":
    main()

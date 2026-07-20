#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监测系统快速测试脚本

测试各个模块是否正常工作
"""

import sys
import os
from pathlib import Path

# Windows命令行UTF-8支持
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time


def test_imports():
    """测试1：检查所有依赖是否可导入"""
    print("=" * 70)
    print("测试 1/5: 检查依赖...")
    print("=" * 70)

    required_modules = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("scipy", "SciPy"),
    ]

    optional_modules = [
        ("funasr", "FunASR (SenseVoice)"),
        ("pyaudio", "PyAudio (麦克风采集)"),
        ("cv2", "OpenCV (RTSP流)"),
    ]

    all_ok = True

    # 必需依赖
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name} - {e}")
            all_ok = False

    # 可选依赖
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} (可选，未安装)")

    return all_ok


def test_audio_stream():
    """测试2：音频流模块"""
    print("\n" + "=" * 70)
    print("测试 2/5: 音频流模块...")
    print("=" * 70)

    try:
        from src.realtime.audio_stream import FileSimulatorStream
        import numpy as np

        # 创建模拟音频数据
        sample_rate = 16000
        duration = 3  # 3秒
        audio_data = np.random.randn(sample_rate * duration).astype(np.float32)

        print(f"  ✅ 音频流模块导入成功")
        print(f"  ✅ 生成测试音频: {len(audio_data)} samples")

        return True
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_sensevoice_engine():
    """测试3：SenseVoice引擎（需要模型）"""
    print("\n" + "=" * 70)
    print("测试 3/5: SenseVoice引擎...")
    print("=" * 70)

    try:
        from src.realtime.sensevoice_engine import RealtimeFeatureAggregator

        # 测试特征聚合器（不依赖模型）
        aggregator = RealtimeFeatureAggregator(window_hours=24)

        # 添加模拟数据
        mock_utterances = [
            {
                "start_sec": 0,
                "duration_sec": 3.2,
                "emotion": "neutral",
                "speech_rate": 4.5,
                "pitch_mean": 215,
            },
            {
                "start_sec": 3.2,
                "duration_sec": 2.8,
                "emotion": "sad",
                "speech_rate": 3.8,
                "pitch_mean": 195,
            },
        ]

        aggregator.add_utterances(mock_utterances, timestamp=time.time())
        features = aggregator.get_current_features()

        print(f"  ✅ 特征聚合器初始化成功")
        print(f"  ✅ 累积特征: {features['n_utterances']} 段语音")
        print(f"     - 悲伤占比: {features['sad_ratio']:.3f}")
        print(f"     - 平均语速: {features['avg_speed']:.2f} 字/秒")

        # 尝试加载完整引擎（可能失败）
        try:
            from src.realtime.sensevoice_engine import SenseVoiceEngine
            print(f"  ⚠️  SenseVoice完整引擎需要下载模型（约1GB）")
            print(f"     首次运行会自动下载，需要稳定网络连接")
        except Exception as e:
            print(f"  ⚠️  完整引擎加载失败（正常，需要GPU和模型）: {e}")

        return True

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitor_initialization():
    """测试4：监测控制器初始化"""
    print("\n" + "=" * 70)
    print("测试 4/5: 监测控制器...")
    print("=" * 70)

    try:
        # 导入但不实际启动
        from src.realtime.monitor import RealtimeMonitor

        print(f"  ✅ 监测控制器模块导入成功")
        print(f"  ℹ️  完整启动需要:")
        print(f"     1. 音频输入源（麦克风/RTSP/文件）")
        print(f"     2. SenseVoice模型（首次会自动下载）")
        print(f"     3. CUDA/CPU推理环境")

        return True

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """测试5：端到端集成测试（模拟）"""
    print("\n" + "=" * 70)
    print("测试 5/5: 端到端模拟...")
    print("=" * 70)

    try:
        import json
        from src.realtime.sensevoice_engine import RealtimeFeatureAggregator

        # 模拟一天的监测数据
        aggregator = RealtimeFeatureAggregator(window_hours=24)

        print("  模拟24小时监测数据...")

        # 模拟不同时段的语音片段
        scenarios = [
            ("早晨", "neutral", 4.5, 215, 2),
            ("上午", "happy", 4.8, 220, 5),
            ("中午", "neutral", 4.2, 210, 3),
            ("下午", "sad", 3.5, 190, 4),  # 异常
            ("傍晚", "sad", 3.3, 185, 3),  # 持续异常
            ("晚上", "neutral", 4.0, 205, 2),
        ]

        base_time = time.time() - 86400  # 24小时前

        for i, (period, emotion, speed, pitch, n_utterances) in enumerate(scenarios):
            timestamp = base_time + (i * 4 * 3600)  # 每4小时

            mock_utterances = [
                {
                    "start_sec": j * 10,
                    "duration_sec": 3.0,
                    "emotion": emotion,
                    "speech_rate": speed,
                    "pitch_mean": pitch,
                }
                for j in range(n_utterances)
            ]

            aggregator.add_utterances(mock_utterances, timestamp)

        # 获取24小时特征
        features = aggregator.get_current_features()

        print(f"\n  ✅ 24小时特征统计:")
        print(f"     - 语音片段数: {features['n_utterances']}")
        print(f"     - 总时长: {features['total_duration']:.1f} 秒")
        print(f"     - 悲伤占比: {features['sad_ratio']:.3f}")
        print(f"     - 平均语速: {features['avg_speed']:.2f} 字/秒")
        print(f"     - 痛苦事件: {features['distress_events']} 次")

        # 简单风险判定
        risk_level = 0
        if features['sad_ratio'] > 0.25:
            risk_level = 3
        elif features['sad_ratio'] > 0.20:
            risk_level = 2
        elif features['sad_ratio'] > 0.15:
            risk_level = 1

        risk_names = ["正常", "关注", "提醒", "严重"]
        print(f"\n  🔍 风险评估: {risk_names[risk_level]} (等级 {risk_level})")

        # 保存测试结果
        test_output = project_root / "data" / "realtime" / "test"
        test_output.mkdir(parents=True, exist_ok=True)

        result_file = test_output / "test_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "features": features,
                "risk_level": risk_level,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  ✅ 测试结果已保存: {result_file}")

        return True

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("实时心理健康监测系统 - 快速测试")
    print("=" * 70)
    print()

    results = []

    # 运行所有测试
    results.append(("依赖检查", test_imports()))
    results.append(("音频流模块", test_audio_stream()))
    results.append(("SenseVoice引擎", test_sensevoice_engine()))
    results.append(("监测控制器", test_monitor_initialization()))
    results.append(("端到端模拟", test_integration()))

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！系统可以正常使用。")
        print("\n下一步:")
        print("  1. 安装音频依赖: pip install funasr modelscope")
        print("  2. 运行完整监测: python scripts/start_realtime_monitor.py --file test_audio.mp3")
        print("  3. 查看使用文档: README_REALTIME.md")
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装。")
        print("\n安装命令:")
        print("  pip install -r requirements.txt")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

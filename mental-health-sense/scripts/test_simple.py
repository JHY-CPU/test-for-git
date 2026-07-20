# -*- coding: utf-8 -*-
"""
实时监测系统测试 - Windows兼容版本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import json


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_dependencies():
    """测试依赖包"""
    print_section("测试 1: 检查依赖包")

    results = {}

    # 核心依赖
    core_deps = [
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "scipy",
    ]

    for dep in core_deps:
        try:
            if dep == "sklearn":
                __import__("sklearn")
            else:
                __import__(dep)
            print(f"  [OK] {dep}")
            results[dep] = True
        except ImportError:
            print(f"  [FAIL] {dep}")
            results[dep] = False

    # 实时监测依赖（可选）
    optional_deps = [
        ("funasr", "SenseVoice模型"),
        ("pyaudio", "麦克风采集"),
        ("cv2", "OpenCV"),
    ]

    print("\n可选依赖:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"  [OK] {name}")
            results[module] = True
        except ImportError:
            print(f"  [WARN] {name} (未安装)")
            results[module] = False

    return results


def test_project_structure():
    """测试项目结构"""
    print_section("测试 2: 检查项目结构")

    required_dirs = [
        "src/realtime",
        "src/data_pipeline",
        "src/baseline",
        "config",
        "scripts",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [FAIL] {dir_path} 不存在")
            all_exist = False

    return all_exist


def test_realtime_modules():
    """测试实时监测模块"""
    print_section("测试 3: 实时监测模块")

    try:
        # 测试音频流模块
        from src.realtime.audio_stream import AudioStream
        print("  [OK] 音频流基类")

        # 测试特征聚合器
        from src.realtime.sensevoice_engine import RealtimeFeatureAggregator
        aggregator = RealtimeFeatureAggregator(window_hours=24)
        print("  [OK] 特征聚合器")

        # 添加测试数据
        mock_data = [
            {
                "start_sec": 0,
                "duration_sec": 3.0,
                "emotion": "neutral",
                "speech_rate": 4.5,
                "pitch_mean": 215,
            },
            {
                "start_sec": 3,
                "duration_sec": 2.5,
                "emotion": "sad",
                "speech_rate": 3.8,
                "pitch_mean": 195,
            },
        ]

        aggregator.add_utterances(mock_data, time.time())
        features = aggregator.get_current_features()

        print(f"  [OK] 特征计算成功")
        print(f"       - 语音片段: {features['n_utterances']}")
        print(f"       - 悲伤占比: {features['sad_ratio']:.3f}")
        print(f"       - 平均语速: {features['avg_speed']:.2f}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_simulation():
    """测试端到端模拟"""
    print_section("测试 4: 端到端模拟")

    try:
        from src.realtime.sensevoice_engine import RealtimeFeatureAggregator

        aggregator = RealtimeFeatureAggregator(window_hours=24)

        print("  模拟一天的监测数据...")

        # 模拟不同时段
        scenarios = [
            ("早晨 08:00", "neutral", 4.5, 215),
            ("上午 10:00", "happy", 4.8, 220),
            ("中午 12:00", "neutral", 4.2, 210),
            ("下午 14:00", "sad", 3.5, 190),  # 异常
            ("下午 16:00", "sad", 3.3, 185),  # 持续
            ("晚上 20:00", "neutral", 4.0, 205),
        ]

        base_time = time.time() - 86400

        for i, (period, emotion, speed, pitch) in enumerate(scenarios):
            timestamp = base_time + (i * 4 * 3600)

            utterances = [
                {
                    "start_sec": j * 10,
                    "duration_sec": 3.0,
                    "emotion": emotion,
                    "speech_rate": speed,
                    "pitch_mean": pitch,
                }
                for j in range(3)
            ]

            aggregator.add_utterances(utterances, timestamp)
            print(f"    {period}: {emotion} (语速{speed})")

        # 获取24小时统计
        features = aggregator.get_current_features()

        print(f"\n  24小时统计:")
        print(f"    - 总片段: {features['n_utterances']}")
        print(f"    - 总时长: {features['total_duration']:.1f}秒")
        print(f"    - 悲伤占比: {features['sad_ratio']:.3f}")
        print(f"    - 平均语速: {features['avg_speed']:.2f}字/秒")
        print(f"    - 痛苦事件: {features['distress_events']}次")

        # 风险判定
        if features['sad_ratio'] > 0.25:
            risk = "严重"
        elif features['sad_ratio'] > 0.20:
            risk = "提醒"
        elif features['sad_ratio'] > 0.15:
            risk = "关注"
        else:
            risk = "正常"

        print(f"\n  风险评估: {risk}")

        # 保存结果
        output_dir = project_root / "data" / "realtime" / "test"
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / "test_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "features": features,
                "risk_level": risk,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  [OK] 测试结果已保存: {result_file.relative_to(project_root)}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("实时心理健康监测系统 - 快速测试")
    print("=" * 60)

    results = []

    # 运行测试
    dep_results = test_dependencies()
    results.append(("依赖检查", all(dep_results.values())))

    results.append(("项目结构", test_project_structure()))
    results.append(("实时模块", test_realtime_modules()))
    results.append(("端到端模拟", test_integration_simulation()))

    # 汇总
    print_section("测试结果汇总")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n所有测试通过！")
        print("\n下一步:")
        print("  1. 安装实时模块依赖:")
        print("     pip install funasr modelscope")
        print("  2. 运行完整监测:")
        print("     python scripts/start_realtime_monitor.py --file test.mp3")
        print("  3. 查看文档:")
        print("     README_REALTIME.md")
    else:
        print("\n部分测试失败，请检查:")
        print("  pip install -r requirements.txt")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

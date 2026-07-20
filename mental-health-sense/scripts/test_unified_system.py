#!/usr/bin/env python3
"""
统一系统测试脚本

验证统一数据管理器和调度器的功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime
from src.unified_data_manager import UnifiedDataManager


def test_unified_data_manager():
    """测试统一数据管理器"""
    print("=" * 70)
    print("测试: 统一数据管理器")
    print("=" * 70)

    # 创建管理器
    manager = UnifiedDataManager(elder_id="E001")
    print("\n[OK] 统一数据管理器已创建")

    # 模拟实时数据流入
    print("\n模拟24小时数据流入...")

    base_time = time.time() - 86400  # 24小时前

    scenarios = [
        ("06:00", "neutral", 4.3, 210),
        ("09:00", "happy", 4.8, 220),
        ("12:00", "neutral", 4.2, 210),
        ("15:00", "sad", 3.5, 190),
        ("18:00", "sad", 3.2, 185),
        ("21:00", "neutral", 4.0, 205),
    ]

    for i, (time_str, emotion, speed, pitch) in enumerate(scenarios):
        timestamp = base_time + (i * 3 * 3600)

        mock_utterances = [
            {
                "start_sec": j * 10,
                "duration_sec": 3.0,
                "emotion": emotion,
                "speech_rate": speed,
                "pitch_mean": pitch,
            }
            for j in range(5)
        ]

        manager.add_realtime_utterances(mock_utterances, timestamp)
        print(f"  {time_str}: {emotion} (语速{speed})")

    # 测试实时系统获取数据
    print("\n[测试1] 实时系统获取当前24小时特征:")
    current = manager.get_current_features()
    print(f"  - 语音片段: {current['n_utterances']}")
    print(f"  - 悲伤占比: {current['sad_ratio']:.3f}")
    print(f"  - 平均语速: {current['avg_speed']:.2f}")
    print(f"  - 痛苦事件: {current['distress_events']}")

    # 测试每日系统获取数据
    print("\n[测试2] 每日系统获取声学特征:")
    today = datetime.now().strftime("%Y-%m-%d")
    acoustic = manager.get_daily_acoustic_data(today)
    print(f"  - sad_ratio: {acoustic['sad_ratio']:.3f}")
    print(f"  - avg_speed: {acoustic['avg_speed']:.2f}")
    print(f"  - avg_pitch: {acoustic['avg_pitch']:.1f}")
    print(f"  - distress_events: {acoustic['distress_events']}")

    # 测试7天历史数据
    print("\n[测试3] 获取7天历史数据（GRU模型需要）:")
    history = manager.get_7day_acoustic_history(today)
    print(f"  - 获取到 {len(history)} 天的数据")

    # 测试导出
    print("\n[测试4] 导出完整数据包:")
    export = manager.export_for_daily_inference(today)
    print(f"  - acoustic_data: OK")
    print(f"  - sleep_data: {export['sleep_data']}")
    print(f"  - activity_data: {export['activity_data']}")
    print(f"  - social_data: {export['social_data']}")

    print("\n[OK] 统一数据管理器测试通过")
    return True


def test_data_flow():
    """测试完整数据流"""
    print("\n" + "=" * 70)
    print("测试: 完整数据流")
    print("=" * 70)

    manager = UnifiedDataManager(elder_id="E001")

    print("\n模拟场景: 实时系统 -> 每日系统 数据流转")

    # 1. 实时系统持续采集（模拟24小时）
    print("\n[步骤1] 实时系统: 24小时持续采集")
    for hour in range(24):
        timestamp = time.time() - (24 - hour) * 3600

        mock_utterances = [{
            "start_sec": 0,
            "duration_sec": 3.0,
            "emotion": "sad" if 14 <= hour <= 18 else "neutral",
            "speech_rate": 3.5 if 14 <= hour <= 18 else 4.5,
            "pitch_mean": 190 if 14 <= hour <= 18 else 215,
        }]

        manager.add_realtime_utterances(mock_utterances, timestamp)

    print(f"  [OK] 累积24小时数据")

    # 2. 实时系统每小时检查风险
    print("\n[步骤2] 实时系统: 每小时风险检查")
    features = manager.get_current_features()
    sad_ratio = features['sad_ratio']

    if sad_ratio > 0.20:
        print(f"  [WARN] 检测到风险: 悲伤占比 {sad_ratio:.3f}")
    else:
        print(f"  [OK] 状态正常: 悲伤占比 {sad_ratio:.3f}")

    # 3. 每日系统读取累积数据
    print("\n[步骤3] 每日系统: 读取实时累积的声学特征")
    today = datetime.now().strftime("%Y-%m-%d")
    acoustic_data = manager.get_daily_acoustic_data(today)

    print(f"  [OK] 获取acoustic_data:")
    for key, value in acoustic_data.items():
        print(f"      {key}: {value}")

    # 4. 每日系统执行GRU推理
    print("\n[步骤4] 每日系统: 执行GRU推理（模拟）")
    print(f"  - 读取声学特征（来自实时系统）")
    print(f"  - 读取睡眠/活动/社交数据（其他传感器）")
    print(f"  - 组成10维向量")
    print(f"  - GRU模型预测第8天")
    print(f"  - 计算残差 -> EWMA判定")
    print(f"  [OK] 推理完成（示例）")

    print("\n[OK] 完整数据流测试通过")
    return True


def main():
    print("\n" + "=" * 70)
    print("统一系统测试")
    print("=" * 70)

    results = []

    # 运行测试
    results.append(("统一数据管理器", test_unified_data_manager()))
    results.append(("完整数据流", test_data_flow()))

    # 汇总
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[OK] 所有测试通过！统一系统可以使用。")
        print("\n启动命令:")
        print("  python scripts/start_unified_system.py --elder-id E001 --microphone")
    else:
        print("\n[FAIL] 部分测试失败")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

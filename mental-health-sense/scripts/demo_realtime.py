"""
快速演示脚本 - 展示实时监测系统的核心功能

无需真实音频输入，纯模拟演示
"""

import sys
import os
from pathlib import Path

# Windows UTF-8支持
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, '')

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
from src.realtime.sensevoice_engine import RealtimeFeatureAggregator


def simulate_24h_monitoring():
    """模拟24小时实时监测"""

    print("=" * 70)
    print("实时心理健康监测系统 - 演示")
    print("=" * 70)
    print("\n模拟场景：老人E001的24小时语音监测\n")

    # 创建特征聚合器
    aggregator = RealtimeFeatureAggregator(window_hours=24)

    # 模拟一天的语音数据
    scenarios = [
        ("06:00", "Morning", "neutral", 4.3, 210, "Normal"),
        ("08:00", "Breakfast", "happy", 4.8, 220, "Good mood"),
        ("10:00", "Watching TV", "neutral", 4.5, 215, "Stable"),
        ("12:00", "Lunch", "neutral", 4.2, 208, "Normal"),
        ("14:00", "Afternoon", "sad", 3.5, 190, "Low mood [!]"),
        ("16:00", "Late PM", "sad", 3.2, 185, "Still low [!]"),
        ("18:00", "Dinner", "sad", 3.4, 188, "Still low [!]"),
        ("20:00", "Evening", "neutral", 4.0, 205, "Better"),
        ("22:00", "Bedtime", "neutral", 4.1, 207, "Stable"),
    ]

    base_time = time.time() - 86400  # 24小时前开始

    print("时间轴：")
    print("-" * 70)

    for i, (time_str, activity, emotion, speed, pitch, status) in enumerate(scenarios):
        timestamp = base_time + (i * 2 * 3600)  # 每2小时

        # 生成该时段的语音片段
        n_utterances = 5  # 每个时段5段对话
        utterances = [
            {
                "start_sec": j * 10,
                "duration_sec": 3.0,
                "emotion": emotion,
                "speech_rate": speed,
                "pitch_mean": pitch,
            }
            for j in range(n_utterances)
        ]

        aggregator.add_utterances(utterances, timestamp)

        # 打印时间线（避免emoji在Windows命令行显示问题）
        emotion_symbol = {
            "happy": "[+]",
            "neutral": "[o]",
            "sad": "[-]",
        }

        print(f"  {time_str} | {activity:12} | {emotion_symbol[emotion]} {emotion:7} | "
              f"Speed{speed:.1f} | {status}")

    print("-" * 70)

    # 获取24小时统计
    features = aggregator.get_current_features()

    print("\n[24h Stats] 24 Hour Statistics:")
    print(f"  总语音片段: {features['n_utterances']} 段")
    print(f"  总时长: {features['total_duration']:.1f} 秒")
    print(f"  悲伤占比: {features['sad_ratio']:.3f} ({features['sad_ratio']*100:.1f}%)")
    print(f"  平均语速: {features['avg_speed']:.2f} 字/秒")
    print(f"  平均基频: {features['avg_pitch']:.1f} Hz")
    print(f"  痛苦事件: {features['distress_events']} 次")

    # 分析最近4小时（晚上时段）
    features_4h = aggregator.get_hourly_features(last_n_hours=4)

    print("\n[Trend] Recent 4 Hours:")
    print(f"  悲伤占比: {features_4h['sad_ratio']:.3f}")
    print(f"  平均语速: {features_4h['avg_speed']:.2f} 字/秒")

    # 风险判定
    print("\n[Risk] Risk Assessment:")

    risk_level = 0
    risk_type = "None"

    if features['sad_ratio'] > 0.30 and features['avg_speed'] < 3.5:
        risk_level = 3
        risk_type = "Depression - Critical"
    elif features['sad_ratio'] > 0.25 and features['avg_speed'] < 3.8:
        risk_level = 2
        risk_type = "Depression - Warning"
    elif features['sad_ratio'] > 0.20 or features['distress_events'] > 3:
        risk_level = 1
        risk_type = "Depression - Attention"

    risk_colors = ["[OK] Normal", "[!] Attention", "[!!] Warning", "[!!!] Critical"]

    print(f"  风险等级: {risk_colors[risk_level]} (Level {risk_level})")
    print(f"  风险类型: {risk_type}")

    # 分析原因
    if risk_level >= 2:
        print("\n[Analysis] Risk Analysis:")
        print(f"  1. Continued sad emotion during 14:00-18:00")
        print(f"  2. Speech rate dropped from 4.5 to 3.2 chars/sec (-28%)")
        print(f"  3. Pitch dropped from 215Hz to 185Hz (-14%)")
        print(f"  4. Sad ratio reached {features['sad_ratio']*100:.1f}% (normal <15%)")

        print("\n[Action] Recommended Actions:")
        if risk_level == 3:
            print("  - Push notification to family (immediate)")
            print("  - Alert community worker")
            print("  - Suggest visit as soon as possible")
        else:
            print("  - Push notification to family (warning mode)")
            print("  - Suggest phone call tonight")
            print("  - Continue monitoring, escalate if persists 3+ days")

    # 保存演示结果
    output_dir = Path(__file__).parent.parent / "data" / "realtime" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "demo_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "demo_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elder_id": "E001",
            "features_24h": features,
            "features_4h": features_4h,
            "risk_level": risk_level,
            "risk_type": risk_type,
            "scenarios": [
                {"time": t, "activity": a, "emotion": e}
                for t, a, e, _, _, _ in scenarios
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Save] Demo result saved: {result_file.relative_to(Path(__file__).parent.parent)}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("\nIn production, this will be replaced by:")
    print("  - Real audio stream (microphone/RTSP)")
    print("  - SenseVoice real-time inference")
    print("  - Full GRU model risk assessment")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    simulate_24h_monitoring()

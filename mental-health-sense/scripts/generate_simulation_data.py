"""
50天模拟数据生成器

生成5位老人的模拟传感器数据，用于端到端测试。
每位老人有不同的基线特征和注入异常模式。

Usage:
    python scripts/generate_simulation_data.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import FULL_FEATURE_NAMES


# ===== 老人配置 =====
ELDER_CONFIGS = {
    "E001": {
        "name": "活跃开朗型",
        "baseline": {
            "sad_ratio": (0.05, 0.02),
            "avg_speed": (4.5, 0.3),
            "avg_pitch": (220, 15),
            "distress_events": (0.1, 0.2),
            "sleep_efficiency": (0.88, 0.04),
            "deep_sleep_ratio": (0.30, 0.03),
            "sfi": (5.0, 1.0),
            "hrv_rmssd": (50, 5),
            "daily_activity": (6000, 800),
            "social_turns": (35, 5),
        },
        "anomaly": {
            "start_day": 25,
            "end_day": 30,
            "features": {
                "sad_ratio": 0.20,
                "avg_speed": 2.5,
                "distress_events": 3.0,
            },
        },
        "description": "Day 25-30 注入抑郁特征（sad_ratio↑ + avg_speed↓ + distress_events↑）",
    },
    "E002": {
        "name": "安静规律型",
        "baseline": {
            "sad_ratio": (0.03, 0.02),
            "avg_speed": (3.8, 0.2),
            "avg_pitch": (200, 10),
            "distress_events": (0.05, 0.1),
            "sleep_efficiency": (0.92, 0.03),
            "deep_sleep_ratio": (0.35, 0.03),
            "sfi": (3.0, 0.8),
            "hrv_rmssd": (55, 5),
            "daily_activity": (4000, 500),
            "social_turns": (20, 4),
        },
        "anomaly": {
            "start_day": 20,
            "end_day": 20,
            "features": {"distress_events": 15.0},
        },
        "description": "Day 20 distress_events 飙升（趋势检测：单日不触发，连续才报警）",
    },
    "E003": {
        "name": "正常波动型",
        "baseline": {
            "sad_ratio": (0.04, 0.03),
            "avg_speed": (4.2, 0.4),
            "avg_pitch": (210, 15),
            "distress_events": (0.1, 0.2),
            "sleep_efficiency": (0.85, 0.06),
            "deep_sleep_ratio": (0.28, 0.04),
            "sfi": (4.5, 1.2),
            "hrv_rmssd": (48, 6),
            "daily_activity": (5000, 1000),
            "social_turns": (30, 8),
        },
        "anomaly": None,
        "description": "无反常态，仅自然波动",
    },
    "E004": {
        "name": "设备故障型",
        "baseline": {
            "sad_ratio": (0.06, 0.03),
            "avg_speed": (4.0, 0.3),
            "avg_pitch": (215, 12),
            "distress_events": (0.15, 0.3),
            "sleep_efficiency": (0.82, 0.05),
            "deep_sleep_ratio": (0.25, 0.04),
            "sfi": (5.5, 1.5),
            "hrv_rmssd": (42, 5),
            "daily_activity": (4500, 700),
            "social_turns": (25, 5),
        },
        "anomaly": {
            "start_day": 10,
            "end_day": 12,
            "type": "missing_data",
        },
        "description": "Day 10-12 连续数据缺失，触发设备离线告警",
    },
    "E005": {
        "name": "缓慢衰退型",
        "baseline": {
            "sad_ratio": (0.04, 0.02),
            "avg_speed": (4.3, 0.3),
            "avg_pitch": (205, 12),
            "distress_events": (0.1, 0.2),
            "sleep_efficiency": (0.86, 0.04),
            "deep_sleep_ratio": (0.30, 0.03),
            "sfi": (4.0, 1.0),
            "hrv_rmssd": (50, 5),
            "daily_activity": (5500, 800),
            "social_turns": (30, 5),
        },
        "anomaly": {
            "start_day": 30,
            "end_day": 50,
            "type": "drift",
            "drift_features": {
                "social_turns": -0.5,
                "daily_activity": -50,
            },
        },
        "description": "Day 30 起社交互动和活动量持续下降",
    },
}

# 特征列表（10维健康特征）
HEALTH_FEATURES = [
    "sad_ratio", "avg_speed", "avg_pitch", "distress_events",
    "sleep_efficiency", "deep_sleep_ratio", "sfi", "hrv_rmssd",
    "daily_activity", "social_turns",
]


def generate_daily_vector(
    day: int,
    elder_config: dict,
    seed: int = 42,
) -> np.ndarray:
    """
    生成单日10维健康特征向量。

    Args:
        day: 第N天 (1-based)
        elder_config: 老人配置
        seed: 随机种子（保证可复现）

    Returns:
        (10,) numpy数组
    """
    rng = np.random.RandomState((seed + day * 13 + abs(hash(elder_config.get("name", "")))) % (2**31))

    baseline = elder_config["baseline"]
    anomaly = elder_config.get("anomaly")

    vector = np.zeros(10, dtype=np.float64)
    missing_mask = np.zeros(10, dtype=bool)

    for i, feat in enumerate(HEALTH_FEATURES):
        if feat in baseline:
            mean, std = baseline[feat]
            value = rng.normal(mean, std)

            # 注入异常
            if anomaly:
                if anomaly.get("type") == "missing_data":
                    if anomaly["start_day"] <= day <= anomaly["end_day"]:
                        # 随机缺失部分特征
                        if rng.random() < 0.5:
                            value = np.nan
                            missing_mask[i] = True

                elif anomaly.get("type") == "drift":
                    if day >= anomaly["start_day"]:
                        drift_features = anomaly.get("drift_features", {})
                        if feat in drift_features:
                            days_drifted = day - anomaly["start_day"] + 1
                            drift_amount = drift_features[feat] * days_drifted
                            value += drift_amount
                            value += rng.normal(0, abs(drift_features[feat]) * 2)

                elif anomaly["start_day"] <= day <= anomaly["end_day"]:
                    anomaly_features = anomaly.get("features", {})
                    if feat in anomaly_features:
                        anomaly_mean = anomaly_features[feat]
                        value = rng.normal(anomaly_mean, abs(anomaly_mean) * 0.3)

            # 约束范围
            if not missing_mask[i]:
                if feat not in ("avg_pitch", "hrv_rmssd", "sfi"):
                    value = max(0.0, value)
                if feat in ("sad_ratio", "sleep_efficiency", "deep_sleep_ratio"):
                    value = min(max(value, 0.0), 1.0)

            vector[i] = value

    return vector


def compute_time_features(day: int, year: int = 2026) -> tuple:
    """计算时间编码 (day_sin, day_cos)"""
    theta = 2 * np.pi * day / 365
    return np.sin(theta), np.cos(theta)


def generate_all_data(
    output_dir: str | Path,
    n_days: int = 50,
    start_date: str = "2026-07-01",
):
    """
    生成所有老人的50天特征数据。

    数据存储结构：
        data/features/{elder_id}/features.csv

    同时也生成原始传感器数据：
        data/raw/{sleep|activity|social|acoustic}/{elder_id}/{date}.json
    """
    output_dir = Path(output_dir)
    features_dir = output_dir / "data" / "features"
    raw_dir = output_dir / "data" / "raw"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    print(f"生成模拟数据: {len(ELDER_CONFIGS)}位老人, {n_days}天")
    print(f"起始日期: {start_date}")
    print(f"输出目录: {output_dir}")

    for elder_id, config in ELDER_CONFIGS.items():
        print(f"\n{'=' * 50}")
        print(f"生成 {elder_id} ({config['name']})")
        print(f"描述: {config['description']}")

        elder_features_dir = features_dir / elder_id
        elder_features_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        for day in range(1, n_days + 1):
            date_dt = start_dt + timedelta(days=day - 1)
            date_str = date_dt.strftime("%Y-%m-%d")

            # 生成健康特征
            health_vec = generate_daily_vector(day, config, seed=hash(elder_id) % 10000)
            day_sin, day_cos = compute_time_features(day)

            # 完整12维向量
            full_vec = np.concatenate([health_vec, [day_sin, day_cos]])

            # 统计缺失
            missing_count = int(np.isnan(health_vec).sum())
            if missing_count >= 3:
                data_quality = "insufficient"
            elif missing_count > 0:
                data_quality = "valid"  # 少量缺失但仍标记为valid（会用前向填充）
            else:
                data_quality = "valid"

            # 构建CSV行
            row = {
                "date": date_str,
                "missing_count": missing_count,
                "data_quality": data_quality,
            }
            for i, name in enumerate(FULL_FEATURE_NAMES):
                row[name] = float(full_vec[i]) if not np.isnan(full_vec[i]) else ""

            rows.append(row)

            # 生成模拟原始数据（JSON格式）
            _generate_raw_data(raw_dir, elder_id, date_str, health_vec, config)

        # 写入CSV
        import pandas as pd
        df = pd.DataFrame(rows)

        # 列顺序
        cols = ["date"] + FULL_FEATURE_NAMES + ["missing_count", "data_quality"]
        df = df[cols]

        csv_path = elder_features_dir / "features.csv"
        df.to_csv(csv_path, index=False)
        print(f"  └─ 特征数据: {csv_path} ({len(df)}行)")

        # 统计
        valid_days = (df["data_quality"] == "valid").sum()
        insufficient_days = (df["data_quality"] == "insufficient").sum()
        print(f"  └─ 有效天: {valid_days}, 数据不足: {insufficient_days}")

    # 保存老人配置
    config_path = output_dir / "data" / "elder_configs.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: {"name": v["name"], "description": v["description"]}
             for k, v in ELDER_CONFIGS.items()},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n老人配置已保存: {config_path}")

    print(f"\n{'=' * 50}")
    print("✅ 全部模拟数据生成完成！")


def _generate_raw_data(
    raw_dir: Path,
    elder_id: str,
    date_str: str,
    health_vec: np.ndarray,
    config: dict,
):
    """生成原始传感器数据（JSON格式，用于数据管道测试）"""
    # 睡眠数据
    sleep_dir = raw_dir / "sleep" / elder_id
    sleep_dir.mkdir(parents=True, exist_ok=True)
    sleep_data = {
        "sleep_efficiency": float(health_vec[4]) if not np.isnan(health_vec[4]) else None,
        "deep_sleep_ratio": float(health_vec[5]) if not np.isnan(health_vec[5]) else None,
        "sfi": float(health_vec[6]) if not np.isnan(health_vec[6]) else None,
        "hrv_rmssd": float(health_vec[7]) if not np.isnan(health_vec[7]) else None,
        "timestamp": f"{date_str}T06:00:00",
    }
    with open(sleep_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(sleep_data, f, ensure_ascii=False)

    # 活动数据
    activity_dir = raw_dir / "activity" / elder_id
    activity_dir.mkdir(parents=True, exist_ok=True)
    activity_data = {
        "daily_activity": float(health_vec[8]) if not np.isnan(health_vec[8]) else None,
        "space_entropy": 2.0 if not np.isnan(health_vec[8]) else None,
        "timestamp": f"{date_str}T23:59:59",
    }
    with open(activity_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(activity_data, f, ensure_ascii=False)

    # 社交数据
    social_dir = raw_dir / "social" / elder_id
    social_dir.mkdir(parents=True, exist_ok=True)
    social_data = {
        "social_turns": float(health_vec[9]) if not np.isnan(health_vec[9]) else None,
        "speech_duration_ratio": 0.12,
        "timestamp": f"{date_str}T23:59:59",
    }
    with open(social_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(social_data, f, ensure_ascii=False)

    # 声学/语义数据
    acoustic_dir = raw_dir / "acoustic" / elder_id
    acoustic_dir.mkdir(parents=True, exist_ok=True)
    acoustic_data = {
        "sad_ratio": float(health_vec[0]) if not np.isnan(health_vec[0]) else None,
        "avg_speed": float(health_vec[1]) if not np.isnan(health_vec[1]) else None,
        "avg_pitch": float(health_vec[2]) if not np.isnan(health_vec[2]) else None,
        "distress_events": float(health_vec[3]) if not np.isnan(health_vec[3]) else None,
        "timestamp": f"{date_str}T23:59:59",
    }
    with open(acoustic_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(acoustic_data, f, ensure_ascii=False)


if __name__ == "__main__":
    # 默认输出到项目根目录
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root

    generate_all_data(output_path, n_days=50, start_date="2026-07-01")

    print("\n下一步:")
    print("  1. 查看生成的数据: ls data/features/*/")
    print("  2. 运行冷启动训练: python -m src.baseline.trainer")
    print("  3. 运行每日推理: python -m src.scheduler.daily_job")

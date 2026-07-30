"""
60天模拟数据生成器（单人系统）

为唯一被监测的老人生成模拟传感器数据，用于端到端测试与演示。
生成的数据落在 data/features/{ELDER_ID}/ 和 data/raw/*/{ELDER_ID}/ 下，
与真实数据入口路径一致——接入真实老人数据时按同样目录结构放入即可。

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

from src.baseline.scaler_utils import FEATURE_NAMES


# ===== 被监测老人 ID（单人系统）=====
# 系统只分析这一位老人。接入真实数据时可沿用此 ID，或改成你自己的编号，
# 只需保证 data/features/{ELDER_ID}/、data/raw/*/{ELDER_ID}/ 目录名一致。
ELDER_ID = "E001"


# ===== 老人配置（单人）=====
# 保留一段注入异常，用于演示"连续偏离才触发预警"的趋势检测能力。
ELDER_CONFIGS = {
    ELDER_ID: {
        "name": "示例老人",
        "baseline": {
            "sleep_efficiency": (0.88, 0.04),
            "deep_sleep_ratio": (0.30, 0.03),
            "sfi": (5.0, 1.0),
            "hrv_rmssd": (50, 5),
            "daily_activity": (6000, 800),
            "social_turns": (35, 5),
        },
        "anomaly": {
            # 异常注入在 Day 40-46：必须避开建档期(1-35天)，
            # 否则异常落在建档期内会被冷启动兜底盖掉，演示看不到预警。
            # 时间线：建档1-35 / 正常36-39 / 异常40-46 / 恢复47-60。
            "start_day": 40,
            "end_day": 46,
            "features": {
                "sleep_efficiency": 0.65,   # 睡眠效率↓
                "deep_sleep_ratio": 0.15,   # 深睡占比↓
                "sfi": 14.0,                # 睡眠碎片化↑
                "hrv_rmssd": 28.0,          # 心率变异↓
            },
        },
        "description": "Day 40-46 注入睡眠恶化特征（sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓），已避开建档期(35天)，用于演示连续偏离触发预警",
    },
}

# 特征列表（6维健康特征，顺序须与 scaler_utils.FEATURE_NAMES 完全一致）
HEALTH_FEATURES = [
    "sleep_efficiency", "deep_sleep_ratio", "sfi", "hrv_rmssd",
    "daily_activity", "social_turns",
]


def generate_daily_vector(
    day: int,
    elder_config: dict,
    seed: int = 42,
) -> np.ndarray:
    """
    生成单日6维健康特征向量。

    Args:
        day: 第N天 (1-based)
        elder_config: 老人配置
        seed: 随机种子（保证可复现）

    Returns:
        (6,) numpy数组
    """
    rng = np.random.RandomState((seed + day * 13 + abs(hash(elder_config.get("name", "")))) % (2**31))

    baseline = elder_config["baseline"]
    anomaly = elder_config.get("anomaly")

    vector = np.zeros(6, dtype=np.float64)
    missing_mask = np.zeros(6, dtype=bool)

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
                if feat not in ("hrv_rmssd", "sfi"):
                    value = max(0.0, value)
                if feat in ("sleep_efficiency", "deep_sleep_ratio"):
                    value = min(max(value, 0.0), 1.0)

            vector[i] = value

    return vector


def generate_all_data(
    output_dir: str | Path,
    n_days: int = 60,
    start_date: str = "2026-07-01",
):
    """
    生成被监测老人的 n_days 天特征数据（默认 60 天）。

    数据存储结构：
        data/features/{elder_id}/features.csv

    同时也生成原始传感器数据：
        data/raw/{sleep|activity|social}/{elder_id}/{date}.json
    """
    output_dir = Path(output_dir)
    features_dir = output_dir / "data" / "features"
    raw_dir = output_dir / "data" / "raw"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    print(f"生成模拟数据: 老人 {ELDER_ID}, {n_days}天")
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

            # 生成健康特征（6维）
            health_vec = generate_daily_vector(day, config, seed=hash(elder_id) % 10000)

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
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = float(health_vec[i]) if not np.isnan(health_vec[i]) else ""

            rows.append(row)

            # 生成模拟原始数据（JSON格式）
            _generate_raw_data(raw_dir, elder_id, date_str, health_vec, config)

        # 写入CSV
        import pandas as pd
        df = pd.DataFrame(rows)

        # 列顺序
        cols = ["date"] + FEATURE_NAMES + ["missing_count", "data_quality"]
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
    print("模拟数据生成完成。")


def _generate_raw_data(
    raw_dir: Path,
    elder_id: str,
    date_str: str,
    health_vec: np.ndarray,
    config: dict,
):
    """生成原始传感器数据（JSON格式，用于数据管道测试）"""
    # 索引对应 HEALTH_FEATURES / FEATURE_NAMES 的 6 维顺序：
    #   0=sleep_efficiency 1=deep_sleep_ratio 2=sfi 3=hrv_rmssd 4=daily_activity 5=social_turns

    # 睡眠数据
    sleep_dir = raw_dir / "sleep" / elder_id
    sleep_dir.mkdir(parents=True, exist_ok=True)
    sleep_data = {
        "sleep_efficiency": float(health_vec[0]) if not np.isnan(health_vec[0]) else None,
        "deep_sleep_ratio": float(health_vec[1]) if not np.isnan(health_vec[1]) else None,
        "sfi": float(health_vec[2]) if not np.isnan(health_vec[2]) else None,
        "hrv_rmssd": float(health_vec[3]) if not np.isnan(health_vec[3]) else None,
        "timestamp": f"{date_str}T06:00:00",
    }
    with open(sleep_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(sleep_data, f, ensure_ascii=False)

    # 活动数据
    activity_dir = raw_dir / "activity" / elder_id
    activity_dir.mkdir(parents=True, exist_ok=True)
    activity_data = {
        "daily_activity": float(health_vec[4]) if not np.isnan(health_vec[4]) else None,
        "space_entropy": 2.0 if not np.isnan(health_vec[4]) else None,
        "timestamp": f"{date_str}T23:59:59",
    }
    with open(activity_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(activity_data, f, ensure_ascii=False)

    # 社交数据
    social_dir = raw_dir / "social" / elder_id
    social_dir.mkdir(parents=True, exist_ok=True)
    social_data = {
        "social_turns": float(health_vec[5]) if not np.isnan(health_vec[5]) else None,
        "speech_duration_ratio": 0.12,
        "timestamp": f"{date_str}T23:59:59",
    }
    with open(social_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(social_data, f, ensure_ascii=False)


if __name__ == "__main__":
    # 默认输出到项目根目录
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root

    generate_all_data(output_path, n_days=60, start_date="2026-07-01")

    print("\n下一步:")
    print("  1. 查看生成的数据: ls data/features/*/")
    print("  2. 运行冷启动训练: python -m src.baseline.trainer")
    print("  3. 运行每日推理: python -m src.scheduler.daily_job")

"""
双轨模拟数据生成器（单人系统）

为唯一被监测的老人生成模拟传感器数据，用于端到端测试与演示。
生成的数据落在 data/features/{ELDER_ID}/ 和 data/raw/*/{ELDER_ID}/ 下，
与真实数据入口路径一致——接入真实老人数据时按同样目录结构放入即可。

时间线（默认 60 天，建档期 35 天）：
    Day  1-35  建档期（两轨都要干净，异常落这里会被学成正常基线）
    Day 36-39  正常过渡
    Day 40-46  ★ 注入睡眠恶化（睡眠轨）
    Day 47-49  恢复
    Day 50-58  ★ 注入社交退缩（社会连接轨）
    Day 59-60  恢复

两段异常刻意错开：这样能验证双轨的信号隔离——睡眠恶化期社交轨应保持正常，
社交退缩期睡眠轨应保持正常。若两轨同时报警，说明残差串轨了。

Usage:
    python scripts/generate_simulation_data.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import SLEEP_FEATURES, SOCIAL_FEATURES, TRACKS
from src.data_pipeline.adapters.circadian import compute_circadian_features
from src.utils.io import get_feature_columns
from src.utils.seeding import stable_hash


ELDER_ID = "E001"

# ===== 基线分布（均值, 标准差）=====
# 数值取真实老人的合理范围：睡眠效率 0.88、夜醒 35 分钟、离床 1.5 次等。
SLEEP_BASELINE = {
    "sleep_efficiency":  (0.88, 0.035),
    "waso_min":          (35.0, 8.0),
    "sol_min":           (18.0, 6.0),
    "bed_exit_count":    (1.5, 0.7),
    "deep_sleep_ratio":  (0.22, 0.03),
    "sleep_onset_clock": (150.0, 25.0),   # 距 20:00 的分钟数 → 约 22:30 入睡
    "night_hr_mean":     (62.0, 4.0),
    "daytime_nap_min":   (40.0, 15.0),
}

# 社交轨的 copresence / out_of_home / activity 直接给分布；
# rar_amplitude / rar_iv 由小时活动序列算出来（不直接采样），
# 这样它们与 activity_counts 的关系是真实派生的，而不是三个独立随机数。
SOCIAL_BASELINE = {
    "copresence_min":  (55.0, 20.0),
    "out_of_home_min": (95.0, 30.0),
    "activity_counts": (185.0, 25.0),
}

# 周末效应：子女探访 → 共处时长与活动量抬升。
# 这是社交轨必须按 is_weekend 分池的原因，模拟数据里必须体现出来，
# 否则测不出"分池 EWMA 减少周末误报"这个收益。
WEEKEND_MULTIPLIER = {
    "copresence_min":  2.4,
    "out_of_home_min": 0.75,   # 有人来访时反而少出门
    "activity_counts": 1.18,
}

# ===== 异常注入 =====
ANOMALY_SLEEP = {
    "start_day": 40,
    "end_day": 46,
    "label": "睡眠恶化",
    "features": {
        "sleep_efficiency": 0.68,   # ↓ 睡眠效率
        "waso_min":         95.0,   # ↑ 夜醒时长
        "sol_min":          48.0,   # ↑ 入睡困难
        "bed_exit_count":   4.2,    # ↑ 频繁起夜
        "deep_sleep_ratio": 0.11,   # ↓ 深睡减少
    },
}

ANOMALY_SOCIAL = {
    "start_day": 50,
    "end_day": 58,
    "label": "社交退缩",
    "features": {
        "copresence_min":  6.0,     # ↓ 几乎没人来
        "out_of_home_min": 12.0,    # ↓ 也不出门
        "activity_counts": 78.0,    # ↓ 家里也不怎么动
    },
    # 节律塌陷：白天活动往夜里挪 + 碎片化，使 RA↓ 且 IV↑
    "circadian_collapse": True,
}


def _rng(day: int, salt: int, seed: int) -> np.random.RandomState:
    """按天+盐值取确定性随机源，保证可复现（用 crc32 而非内置 hash）"""
    return np.random.RandomState((seed + day * 13 + salt) % (2**31))


def _in_range(day: int, anomaly: dict) -> bool:
    return anomaly["start_day"] <= day <= anomaly["end_day"]


def generate_sleep_vector(day: int, seed: int) -> np.ndarray:
    """生成单日 8 维睡眠特征"""
    rng = _rng(day, 101, seed)
    vec = np.zeros(len(SLEEP_FEATURES), dtype=np.float64)

    anomalous = _in_range(day, ANOMALY_SLEEP)

    for i, feat in enumerate(SLEEP_FEATURES):
        mean, std = SLEEP_BASELINE[feat]
        if anomalous and feat in ANOMALY_SLEEP["features"]:
            target = ANOMALY_SLEEP["features"][feat]
            value = rng.normal(target, abs(target) * 0.12)
        else:
            value = rng.normal(mean, std)

        # 物理约束
        if feat in ("sleep_efficiency", "deep_sleep_ratio"):
            value = float(np.clip(value, 0.05, 1.0))
        elif feat == "bed_exit_count":
            value = float(max(0.0, round(value)))
        elif feat == "night_hr_mean":
            value = float(np.clip(value, 40.0, 110.0))
        else:
            value = float(max(0.0, value))

        vec[i] = value

    # 睡眠效率与夜醒时长应当负相关（WASO 多 → SE 低）。
    # 独立采样会丢掉这个关系，让 GRU 学到不存在的独立性。
    tib = 480.0
    implied_se = float(np.clip(1.0 - vec[1] / tib - vec[2] / tib, 0.05, 1.0))
    se_idx = SLEEP_FEATURES.index("sleep_efficiency")
    vec[se_idx] = float(np.clip(0.5 * vec[se_idx] + 0.5 * implied_se, 0.05, 1.0))

    return vec


def generate_hourly_activity(day: int, seed: int, is_weekend: bool) -> np.ndarray:
    """
    生成 24 小时活动序列 a[0..23]。RA/IV 从这里派生。

    正常作息：夜间近零，早晨起床后上升，午后小低谷，傍晚回升，睡前下降。
    节律塌陷：白天压低 + 夜间抬高 + 逐小时噪声加大 → RA↓ 且 IV↑。
    """
    rng = _rng(day, 202, seed)
    collapse = _in_range(day, ANOMALY_SOCIAL) and ANOMALY_SOCIAL.get("circadian_collapse")

    # 基础日间轮廓（下标=小时）
    profile = np.array([
        0.2, 0.1, 0.1, 0.1, 0.2, 0.6,     # 00-05 夜间
        3.0, 8.0, 11.0, 12.0, 10.0, 9.0,  # 06-11 上午活跃
        8.0, 6.0, 5.0, 7.0, 10.0, 12.0,   # 12-17 午后低谷后回升
        11.0, 8.0, 5.0, 2.5, 1.0, 0.4,    # 18-23 傍晚到入睡
    ], dtype=np.float64)

    if is_weekend:
        profile = profile * 1.15

    if collapse:
        # 白天塌一半、夜里抬起来 → M10 降、L5 升 → RA 下降
        day_hours = slice(7, 21)
        profile = profile.copy()
        profile[day_hours] *= 0.35
        profile[0:6] += 1.8
        profile[22:24] += 1.5
        noise_scale = 2.2   # 碎片化：相邻小时差分变大 → IV 上升
    else:
        noise_scale = 0.8

    counts = profile + rng.normal(0, noise_scale, size=24)
    return np.maximum(counts, 0.0)


def generate_social_vector(
    day: int,
    seed: int,
    is_weekend: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成单日 5 维社会连接特征。

    Returns:
        (vector, hourly_counts) —— hourly_counts 一并返回供 raw JSON 落盘
    """
    rng = _rng(day, 303, seed)
    hourly = generate_hourly_activity(day, seed, is_weekend)
    circ = compute_circadian_features(hourly)

    anomalous = _in_range(day, ANOMALY_SOCIAL)

    values: dict[str, float] = {}
    for feat, (mean, std) in SOCIAL_BASELINE.items():
        if anomalous and feat in ANOMALY_SOCIAL["features"]:
            target = ANOMALY_SOCIAL["features"][feat]
            value = rng.normal(target, abs(target) * 0.15)
        else:
            base = mean * WEEKEND_MULTIPLIER.get(feat, 1.0) if is_weekend else mean
            value = rng.normal(base, std)
        values[feat] = float(max(0.0, value))

    # activity_counts 与小时序列保持一致（否则 a[h] 与总数自相矛盾）
    hourly_total = float(hourly.sum())
    if hourly_total > 0:
        scale = values["activity_counts"] / hourly_total
        hourly = hourly * scale

    vec = np.zeros(len(SOCIAL_FEATURES), dtype=np.float64)
    for i, feat in enumerate(SOCIAL_FEATURES):
        if feat == "rar_amplitude":
            vec[i] = circ["rar_amplitude"]
        elif feat == "rar_iv":
            vec[i] = circ["rar_iv"]
        else:
            vec[i] = values[feat]

    return vec, hourly


def generate_all_data(
    output_dir: str | Path,
    n_days: int = 60,
    start_date: str = "2026-07-01",
) -> None:
    """
    生成被监测老人的 n_days 天双轨特征数据。

    存储结构：
        data/features/{elder_id}/features_sleep.csv    8 维
        data/features/{elder_id}/features_social.csv   5 维
        data/raw/{sleep|activity|camera}/{elder_id}/{day_key}.json
    """
    import pandas as pd

    output_dir = Path(output_dir)
    features_dir = output_dir / "data" / "features" / ELDER_ID
    raw_dir = output_dir / "data" / "raw"
    features_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    seed = stable_hash(ELDER_ID) % 10000

    print(f"生成双轨模拟数据: 老人 {ELDER_ID}, {n_days} 天")
    print(f"起始日期: {start_date}")
    print(f"输出目录: {output_dir}")
    print(f"  睡眠轨 {len(SLEEP_FEATURES)} 维 / 社会连接轨 {len(SOCIAL_FEATURES)} 维")
    print(f"  异常段: Day {ANOMALY_SLEEP['start_day']}-{ANOMALY_SLEEP['end_day']} "
          f"{ANOMALY_SLEEP['label']}（睡眠轨）")
    print(f"          Day {ANOMALY_SOCIAL['start_day']}-{ANOMALY_SOCIAL['end_day']} "
          f"{ANOMALY_SOCIAL['label']}（社会连接轨）")

    sleep_rows, social_rows = [], []

    for day in range(1, n_days + 1):
        date_dt = start_dt + timedelta(days=day - 1)
        day_key = date_dt.strftime("%Y-%m-%d")
        is_weekend = date_dt.weekday() >= 5

        sleep_vec = generate_sleep_vector(day, seed)
        social_vec, hourly = generate_social_vector(day, seed, is_weekend)

        sleep_rows.append(_build_row(day_key, sleep_vec, SLEEP_FEATURES))
        social_rows.append(_build_row(day_key, social_vec, SOCIAL_FEATURES))

        _write_raw(raw_dir, day_key, sleep_vec, social_vec, hourly)

    for track, rows, names in (
        ("sleep", sleep_rows, SLEEP_FEATURES),
        ("social", social_rows, SOCIAL_FEATURES),
    ):
        df = pd.DataFrame(rows)[get_feature_columns(track)]
        csv_path = features_dir / f"features_{track}.csv"
        df.to_csv(csv_path, index=False)
        valid = int((df["data_quality"] == "valid").sum())
        print(f"  └─ {csv_path.name}: {len(df)} 行，valid {valid} 天")

    config_path = output_dir / "data" / "elder_configs.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            ELDER_ID: {
                "name": "示例老人",
                "tracks": list(TRACKS),
                "sleep_features": SLEEP_FEATURES,
                "social_features": SOCIAL_FEATURES,
                "anomalies": [
                    {
                        "track": "sleep",
                        "label": ANOMALY_SLEEP["label"],
                        "start_day": ANOMALY_SLEEP["start_day"],
                        "end_day": ANOMALY_SLEEP["end_day"],
                    },
                    {
                        "track": "social",
                        "label": ANOMALY_SOCIAL["label"],
                        "start_day": ANOMALY_SOCIAL["start_day"],
                        "end_day": ANOMALY_SOCIAL["end_day"],
                    },
                ],
                "description": (
                    "两段异常刻意错开，用于验证双轨信号隔离："
                    "睡眠恶化期社交轨应正常，社交退缩期睡眠轨应正常。"
                ),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n老人配置已保存: {config_path}")
    print("模拟数据生成完成。")


def _build_row(day_key: str, vec: np.ndarray, names: list[str]) -> dict:
    """组装一行 CSV 记录。模拟数据不注入缺失，故 missing_count=0 / valid。"""
    row: dict = {name: float(vec[i]) for i, name in enumerate(names)}
    row["day_key"] = day_key
    row["missing_count"] = 0
    row["data_quality"] = "valid"
    return row


def _write_raw(
    raw_dir: Path,
    day_key: str,
    sleep_vec: np.ndarray,
    social_vec: np.ndarray,
    hourly: np.ndarray,
) -> None:
    """
    落原始传感器 JSON，键名与 aggregator 期望的特征名一致。

    分三路对应三类设备：
        sleep    → 小贝壳
        activity → T1C + C6c 事件派生（含小时序列）
        camera   → 边缘人形检测（copresence）
    """
    sleep_payload = {
        name: float(sleep_vec[i]) for i, name in enumerate(SLEEP_FEATURES)
    }
    sleep_payload["timestamp"] = f"{day_key}T06:00:00"

    social_map = {name: float(social_vec[i]) for i, name in enumerate(SOCIAL_FEATURES)}
    activity_payload = {
        "activity_counts": social_map["activity_counts"],
        "out_of_home_min": social_map["out_of_home_min"],
        "rar_amplitude": social_map["rar_amplitude"],
        "rar_iv": social_map["rar_iv"],
        "hourly_activity": [round(float(v), 3) for v in hourly],
        "timestamp": f"{day_key}T23:59:59",
    }
    camera_payload = {
        "copresence_min": social_map["copresence_min"],
        "timestamp": f"{day_key}T23:59:59",
    }

    for subdir, payload in (
        ("sleep", sleep_payload),
        ("activity", activity_payload),
        ("camera", camera_payload),
    ):
        target = raw_dir / subdir / ELDER_ID
        target.mkdir(parents=True, exist_ok=True)
        with open(target / f"{day_key}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_all_data(project_root, n_days=60, start_date="2026-07-01")

    print("\n下一步:")
    print("  1. 查看数据: ls data/features/E001/")
    print("  2. 双轨建档: python scripts/train_all_baselines.py")
    print("  3. 每日推理: python scripts/run_daily_pipeline.py")

"""合成数据判别力验证（范围 2：有效果吗，双轨）

范围 1 只证明"链路跑通"。范围 2 用**更刁钻**的合成数据测判别力：
    - 真阳性(TP)：真异常 → 应检出、且判对类型
    - 真阴性/混淆项(TN)：像异常但不该报 → 应保持安静
    - 已知漏报(KM)：如实记录规则设计带来的灵敏度代价，不算失败

关键改进（打破范围 1 的"循环论证"）：
    1. 正常天带真实噪声：AR(1) 自相关 + 周末效应，不再是纯 iid 高斯
    2. 三类异常各自独立注入，验证三条规则互不误触
    3. 混淆项覆盖真实误报源：单日尖峰、短期社交低、纯周末效应、方向反了

★ 关于"串味"：v2.1 双轨从**结构上**切断了 v2.0 的主要串味源——
两轨各自独立的 GRU 与 scaler，睡眠残差不再经隐藏状态耦合进社交预测。
故本脚本对 TP 场景**同时**断言"应报的类型报了"与"另一轨保持安静"，
这是 v2.0 做不到的（那时只能断言前者）。

指标：TP 检出率 + 判型正确率；TN 特异度。这是 docs/VALIDATION.md 层次 B。
仍**不**证明层次 C（真实有效性需真人 + 临床标签 + 量表对照）。

用独立 elder_id（每场景一个），跑完自动清理。

Usage:
    python scripts/validate_discriminative.py
    python scripts/validate_discriminative.py --only TP_sleep
"""

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import SLEEP_FEATURES, SOCIAL_FEATURES
from src.utils.io import get_project_root, load_config
from src.utils.logger import setup_logger

N_DAYS = 60
BUILD_DAYS = 35          # 与 settings.yaml training.initial.build_days 对齐
START_DATE = "2026-01-01"

# 正常基线（均值, 标准差）——与 generate_simulation_data 的 E001 口径一致
SLEEP_BASELINE = {
    "sleep_efficiency": (0.88, 0.04),
    "waso_min": (38.0, 10.0),
    "sol_min": (18.0, 6.0),
    "bed_exit_count": (1.2, 0.7),
    "deep_sleep_ratio": (0.22, 0.04),
    "sleep_onset_clock": (165.0, 25.0),
    "night_hr_mean": (62.0, 4.0),
    "daytime_nap_min": (25.0, 15.0),
}
SOCIAL_BASELINE = {
    "copresence_min": (75.0, 30.0),
    "out_of_home_min": (95.0, 30.0),
    "rar_amplitude": (0.82, 0.06),
    "rar_iv": (0.55, 0.10),
    "activity_counts": (120.0, 25.0),
}

# 周末效应：子女探访 → 共处涨、外出跌。这是社交轨必须分池的原因。
WEEKEND_FACTORS = {"copresence_min": 2.4, "out_of_home_min": 0.75}

# AR(1) 自相关系数：老人作息有惯性，今天像昨天。纯 iid 会让检出显得过于容易。
AR1_RHO = 0.35

# 比例类特征的合法区间
BOUNDED = {
    "sleep_efficiency": (0.0, 1.0),
    "deep_sleep_ratio": (0.0, 1.0),
    "rar_amplitude": (0.0, 1.0),
    "rar_iv": (0.0, 2.0),
}
NON_NEGATIVE = {
    "waso_min", "sol_min", "bed_exit_count", "daytime_nap_min",
    "copresence_min", "out_of_home_min", "activity_counts",
}


def _gen_series(baseline: dict, order: list[str], seed: int, weekend: bool) -> np.ndarray:
    """生成 (N_DAYS, n_features) 正常序列：AR(1) 噪声 + 可选周末效应。"""
    rng = np.random.default_rng(seed)
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    series = np.zeros((N_DAYS, len(order)))

    for i, name in enumerate(order):
        mean, std = baseline[name]
        noise = np.zeros(N_DAYS)
        for d in range(N_DAYS):
            innov = rng.normal(0, std * np.sqrt(1 - AR1_RHO ** 2))
            noise[d] = AR1_RHO * noise[d - 1] + innov if d else rng.normal(0, std)
        series[:, i] = mean + noise

        if weekend and name in WEEKEND_FACTORS:
            for d in range(N_DAYS):
                if (start + timedelta(days=d)).weekday() >= 5:
                    series[d, i] *= WEEKEND_FACTORS[name]

    return _clip(series, order)


def _clip(series: np.ndarray, order: list[str]) -> np.ndarray:
    """把各维压回物理合法范围。"""
    for i, name in enumerate(order):
        if name in BOUNDED:
            lo, hi = BOUNDED[name]
            series[:, i] = np.clip(series[:, i], lo, hi)
        elif name in NON_NEGATIVE:
            series[:, i] = np.maximum(series[:, i], 0.0)
    return series


def apply_injection(
    sleep: np.ndarray, social: np.ndarray, spec: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """把异常注入两轨序列。spec 里 features 按轨分组。"""
    if not spec:
        return sleep, social

    rng = np.random.default_rng(seed + 991)
    s_idx = {n: i for i, n in enumerate(SLEEP_FEATURES)}
    so_idx = {n: i for i, n in enumerate(SOCIAL_FEATURES)}
    typ = spec.get("type", "block")
    start, end = spec["start"], spec["end"]

    for track, feats in spec.get("features", {}).items():
        arr = sleep if track == "sleep" else social
        idx = s_idx if track == "sleep" else so_idx
        base = SLEEP_BASELINE if track == "sleep" else SOCIAL_BASELINE
        order = SLEEP_FEATURES if track == "sleep" else SOCIAL_FEATURES

        for name, target in feats.items():
            i = idx[name]
            std = base[name][1]
            for day in range(start - 1, min(end, N_DAYS)):
                if typ == "block":
                    arr[day, i] = target + rng.normal(0, std * 0.25)
                elif typ == "drift":
                    frac = (day - start + 2) / (end - start + 1)
                    arr[day, i] = (
                        base[name][0] + (target - base[name][0]) * frac
                        + rng.normal(0, std * 0.4)
                    )
                elif typ == "spike":
                    arr[day, i] = target + rng.normal(0, abs(target) * 0.05 + 1e-6)
        _clip(arr, order)

    return sleep, social


# 场景定义。expect_active/expect_silent 是"应/不应激活的风险类型"。
SCENARIOS = [
    # ---------- 真阳性 TP ----------
    {
        "id": "TP_sleep", "kind": "TP",
        "desc": "睡眠稳定性恶化：效率↓ + WASO↑ + 离床↑，连续 7 天",
        "inject": {
            "type": "block", "start": 40, "end": 46,
            "features": {"sleep": {
                "sleep_efficiency": 0.68, "waso_min": 95.0, "bed_exit_count": 4.5,
            }},
        },
        "expect_active": ["sleep_stability"],
        "expect_silent": ["social_decline"],
        "expect_detect": True,
    },
    {
        "id": "TP_sleep_minimal", "kind": "TP",
        "desc": "只两个必选维恶化（效率↓ + WASO↑），考验必选/可选的设计",
        "inject": {
            "type": "block", "start": 40, "end": 46,
            "features": {"sleep": {"sleep_efficiency": 0.66, "waso_min": 100.0}},
        },
        # 只有 2 必选、0 可选 → 按规则 min_optional=1 不该激活类型，
        # 但异常分仍应越限（检出）。这正是"检出"与"判型"分离的地方。
        "expect_active": [], "expect_silent": ["social_decline"],
        "expect_detect": True,
    },
    {
        "id": "TP_sleep_drift", "kind": "TP",
        "desc": "渐变睡眠恶化：10 天缓慢漂移（考验趋势捕捉，非阶跃）",
        "inject": {
            "type": "drift", "start": 38, "end": 48,
            "features": {"sleep": {
                "sleep_efficiency": 0.62, "waso_min": 110.0,
                "bed_exit_count": 5.0, "sol_min": 55.0,
            }},
        },
        "expect_active": ["sleep_stability"],
        "expect_silent": ["social_decline"],
        "expect_detect": True,
    },
    {
        "id": "TP_social", "kind": "TP",
        "desc": "社会连接减弱：共处↓ + 外出↓ + 活动↓，连续 11 天",
        "inject": {
            "type": "block", "start": 40, "end": 50,
            "features": {"social": {
                "copresence_min": 6.0, "out_of_home_min": 12.0,
                "activity_counts": 45.0,
            }},
        },
        "expect_active": ["social_decline"],
        "expect_silent": ["sleep_stability"],
        "expect_detect": True,
    },
    {
        "id": "TP_circadian", "kind": "TP",
        "desc": "作息节律紊乱：RA↓ + IV↑ + 入睡相位漂移，连续 8 天",
        "inject": {
            "type": "block", "start": 40, "end": 47,
            "features": {
                "social": {"rar_amplitude": 0.35, "rar_iv": 1.35},
                "sleep": {"sleep_onset_clock": 330.0},
            },
        },
        "expect_active": ["circadian_disruption"],
        "expect_silent": [],
        "expect_detect": True,
    },
    # ---------- 真阴性 / 混淆项 TN ----------
    {
        "id": "TN_all_normal", "kind": "TN",
        "desc": "全程正常（AR1 噪声 + 周末效应）：测基础误报率",
        "inject": None,
        "expect_active": [],
        "expect_silent": ["sleep_stability", "social_decline", "circadian_disruption"],
        "expect_detect": False, "expect_max_level": 1,
    },
    {
        "id": "TN_single_spike", "kind": "TN",
        "desc": "单日剧烈波动后恢复：不该升级（持续性门槛应拦住）",
        "inject": {
            "type": "spike", "start": 43, "end": 43,
            "features": {"sleep": {
                "sleep_efficiency": 0.48, "waso_min": 150.0, "bed_exit_count": 7.0,
            }},
        },
        "expect_active": [], "expect_silent": ["sleep_stability"],
        "expect_detect": False, "expect_max_level": 1,
    },
    {
        "id": "TN_short_social", "kind": "TN",
        "desc": "社交低落仅 4 天（<7天窗内5天门槛）→ 不该报社会连接减弱",
        "inject": {
            "type": "block", "start": 41, "end": 44,
            "features": {"social": {
                "copresence_min": 15.0, "out_of_home_min": 30.0,
                "activity_counts": 60.0,
            }},
        },
        "expect_active": [], "expect_silent": ["social_decline"],
        "expect_detect": False, "expect_max_level": 2,
    },
    {
        "id": "TN_sleep_improved", "kind": "TN",
        "desc": "睡眠全面变好（效率↑ WASO↓ 深睡↑ 夜心率↓）→ 方向闸门应封顶",
        # 必须**生理自洽**地改善所有维：只把效率和 WASO 调好、深睡和夜心率留在基线，
        # 得到的是一个现实中不存在的组合，且 GRU 会因输入剧变而预测失准，
        # 在未注入的维上产出朝坏方向的残差假象——那时闸门放行是对的，
        # 测的却不是"好转"。
        "inject": {
            "type": "block", "start": 40, "end": 50,
            "features": {"sleep": {
                "sleep_efficiency": 0.97, "waso_min": 8.0, "bed_exit_count": 0.0,
                "sol_min": 6.0, "deep_sleep_ratio": 0.34, "night_hr_mean": 55.0,
                "daytime_nap_min": 5.0,
            }},
        },
        "expect_active": [], "expect_silent": ["sleep_stability"],
        "expect_detect": False, "expect_max_level": 2,
    },
    # ---------- 已知漏报 KM：如实记录，不算失败 ----------
    {
        "id": "KM_social_partial", "kind": "KM",
        "desc": "只共处↓、外出照常（子女不来但自己照常出门）",
        "inject": {
            "type": "block", "start": 40, "end": 50,
            "features": {"social": {"copresence_min": 4.0}},
        },
        "note": "三项全中的规则必然漏这一类；这是换取低误报的代价，已写入 VALIDATION.md 局限③",
        "expect_active": [], "expect_silent": [],
        "expect_detect": False, "expect_max_level": 3,
    },
]


def cleanup(velder: str):
    root = get_project_root() / "data"
    targets = [root / "features" / velder, root / "baselines" / velder] + [
        root / "raw" / s / velder for s in ("sleep", "activity", "camera")
    ]
    for p in targets:
        shutil.rmtree(p, ignore_errors=True)
    for sub in ("daily_inference", "alerts"):
        d = root / "logs" / sub
        if d.exists():
            for f in d.glob(f"{velder}_*"):
                f.unlink()


def run_scenario(scn: dict, config) -> dict:
    """跑一个场景：造数据 → 逐日双轨管道 → 建档期末训练。"""
    import torch

    import scripts.generate_simulation_data as gen
    from src.baseline.trainer import train_all_tracks
    from src.scheduler.daily_job import load_raw_sensors, run_daily_pipeline

    velder = "D_" + scn["id"]
    cleanup(velder)
    root = get_project_root()

    # 用 hashlib 而非内置 hash()：内置 hash() 受 PYTHONHASHSEED 随机化影响，
    # 每次进程启动结果不同，会让通过数在场景间来回飘。md5 是确定性的。
    seed = int(hashlib.md5(scn["id"].encode()).hexdigest(), 16) % 100000
    # GRU 初始权重也要固定，否则边界场景的通过/不通过会翻转。
    torch.manual_seed(seed)

    sleep = _gen_series(SLEEP_BASELINE, SLEEP_FEATURES, seed, weekend=False)
    social = _gen_series(SOCIAL_BASELINE, SOCIAL_FEATURES, seed + 1, weekend=True)
    sleep, social = apply_injection(sleep, social, scn.get("inject"), seed)

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    original = gen.ELDER_ID
    gen.ELDER_ID = velder
    try:
        for day in range(N_DAYS):
            date_dt = start + timedelta(days=day)
            day_key = date_dt.strftime("%Y-%m-%d")
            # 小时序列只作诊断落盘用（aggregator 直接读 rar_* 标量），
            # 故复用生成器的作息曲线即可，不必反推自注入后的 RA/IV。
            hourly = gen.generate_hourly_activity(
                day + 1, seed, date_dt.weekday() >= 5
            )
            gen._write_raw(
                root / "data" / "raw", day_key, sleep[day], social[day], hourly
            )
    finally:
        gen.ELDER_ID = original

    max_level = 0
    active_types: set[str] = set()
    track_dev = {"sleep": 0, "social": 0}
    try:
        for day in range(1, N_DAYS + 1):
            day_key = (start + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            res = run_daily_pipeline(
                velder, day_key, raw_data=load_raw_sensors(velder, day_key), config=config
            )
            if day == BUILD_DAYS:
                train_all_tracks(velder, config)
                continue
            if day <= BUILD_DAYS:
                continue

            risk = res.get("risk_result") or {}
            max_level = max(max_level, risk.get("risk_level") or 0)
            for rt in risk.get("risk_types", []):
                active_types.add(rt.get("risk_key"))
            inf = res.get("inference_result") or {}
            for track in track_dev:
                if (inf.get(track) or {}).get("is_deviation"):
                    track_dev[track] += 1
    finally:
        cleanup(velder)

    return {"max_level": max_level, "active_types": active_types, "track_dev": track_dev}


def evaluate(scn: dict, obs: dict) -> tuple[bool, str]:
    """对照期望打分。KM 场景只记录，不判失败。"""
    if scn["kind"] == "KM":
        return True, f"已知漏报（预期行为）：最高 L{obs['max_level']}"

    active, lvl = obs["active_types"], obs["max_level"]
    problems = []

    for t in scn.get("expect_active", []):
        if t not in active:
            problems.append(f"应报未报:{t}")
    for t in scn.get("expect_silent", []):
        if t in active:
            problems.append(f"误报:{t}")

    if scn["kind"] == "TN":
        cap = scn.get("expect_max_level", 1)
        if lvl > cap:
            problems.append(f"等级越限:L{lvl}>{cap}")
    if scn.get("expect_detect") and lvl < 2:
        problems.append(f"未检出:最高仅L{lvl}")

    return not problems, "; ".join(problems) if problems else "符合预期"


TYPE_ABBR = {
    "sleep_stability": "睡眠", "social_decline": "社会", "circadian_disruption": "节律",
}


def main():
    parser = argparse.ArgumentParser(description="合成数据判别力验证（范围2，双轨）")
    parser.add_argument("--only", help="只跑指定场景 id")
    args = parser.parse_args()

    setup_logger(log_level="ERROR")
    config = load_config()

    scenarios = [s for s in SCENARIOS if not args.only or s["id"] == args.only]
    if not scenarios:
        print(f"没有匹配 --only {args.only} 的场景")
        return 1

    print("\n" + "=" * 92)
    print("合成数据判别力验证（范围2：有效果吗，双轨）  正常天带 AR1 噪声 + 周末效应")
    print("=" * 92)
    print(f"{'类':>3} {'场景':>20} {'级':>3} {'睡偏':>4} {'社偏':>4} {'激活类型':>14} {'判定':>4}")
    print("-" * 92)

    results, passed, scored = [], 0, 0
    for scn in scenarios:
        obs = run_scenario(scn, config)
        ok, reason = evaluate(scn, obs)
        results.append((scn, obs, ok, reason))
        if scn["kind"] != "KM":
            scored += 1
            passed += ok
        types = "、".join(TYPE_ABBR.get(t, t) for t in sorted(obs["active_types"])) or "无"
        mark = "[v]" if ok else "[x]"
        print(f"{scn['kind']:>3} {scn['id']:>20} {'L'+str(obs['max_level']):>3} "
              f"{obs['track_dev']['sleep']:>4} {obs['track_dev']['social']:>4} "
              f"{types:>12} {mark:>4}")

    print("-" * 92)
    print(f"通过 {passed}/{scored}（KM 场景不计分）")
    print("\n逐场景说明：")
    for scn, obs, ok, reason in results:
        print(f"  {'[v]' if ok else '[x]'} {scn['id']}: {scn['desc']}")
        if scn.get("note"):
            print(f"       └─ 备注: {scn['note']}")
        if not ok:
            print(f"       └─ 问题: {reason}")

    print("=" * 92)
    print("注：范围2 用合成数据测判别力（层次 B）。已加 AR1 噪声/周末效应/混淆项/方向反例，")
    print("    但仍非真人数据——不证明真实有效（层次 C 需临床标签 + 量表对照）。")
    print("    '睡偏/社偏' = 该轨在正式运行期被判偏离的天数，用于看信号是否落在正确的轨。")
    return 0 if passed == scored else 1


if __name__ == "__main__":
    sys.exit(main())

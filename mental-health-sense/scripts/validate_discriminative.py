"""合成数据判别力验证（范围 2：有效果吗）

范围 1 只证明"链路跑通"。范围 2 用**更刁钻**的合成数据测判别力：
    - 真阳性(TP)：真异常 → 应检出、且判对类型
    - 真阴性/混淆项(TN)：像异常但不该报 → 应保持安静

关键改进（打破范围1的"循环论证"）：
    1. 正常天带真实噪声：AR(1) 自相关 + 周节律（周末话少），不再是纯 iid 高斯
    2. 多种异常：典型 / 部分（只2特征）/ 睡眠 / 单日尖峰 / 短时社交低
    3. 混淆项：感冒（睡眠差但情绪正常，不该报抑郁）、短社交低（<5天不该报孤独）、
       单日尖峰（不该升级）

指标：TP 检出率 + 判型正确率；TN 特异度（不误报）。这是 docs/VALIDATION.md 层次 B。
仍**不**证明层次 C（真实有效性需真人+临床标签）。

用独立 elder_id（每场景一个），跑完自动清理。

Usage:
    python scripts/validate_discriminative.py
    python scripts/validate_discriminative.py --keep
"""

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_simulation_data import _generate_raw_data, HEALTH_FEATURES
from src.utils.io import get_project_root, load_config
from src.utils.logger import setup_logger

N_DAYS = 60
BUILD_DAYS = 21
START_DATE = "2026-01-01"

# 正常基线（均值, 标准差）——与前面场景一致
BASELINE = {
    "sad_ratio": (0.05, 0.02), "avg_speed": (4.5, 0.3),
    "pitch_variability": (32, 4), "distress_events": (0.1, 0.2),
    "sleep_efficiency": (0.88, 0.04), "deep_sleep_ratio": (0.30, 0.03),
    "sfi": (5.0, 1.0), "hrv_rmssd": (50, 5),
    "daily_activity": (6000, 800), "social_turns": (35, 5),
}
IDX = {name: i for i, name in enumerate(HEALTH_FEATURES)}

def gen_normal_series(seed: int, n: int = N_DAYS) -> np.ndarray:
    """生成 n 天带真实噪声的正常数据 (n, 10)。

    真实感来自两点（打破纯 iid 高斯）：
      - AR(1) 自相关：今天 = 0.5*偏离(昨天) + 新噪声，模拟"连着几天偏高/偏低"
      - 周节律：周末(第6/7天) social_turns、daily_activity 自然降低
    """
    rng = np.random.RandomState(seed)
    series = np.zeros((n, 10), dtype=np.float64)
    prev_dev = np.zeros(10)
    for day in range(n):
        for name, (mean, std) in BASELINE.items():
            i = IDX[name]
            # AR(1)：保留一半昨日偏离 + 新噪声
            dev = 0.5 * prev_dev[i] + rng.normal(0, std)
            val = mean + dev
            prev_dev[i] = dev
            series[day, i] = val
        # 周节律：周末社交/活动自然下降（day%7 in {5,6}）——正常现象，不该报孤独
        if day % 7 in (5, 6):
            series[day, IDX["social_turns"]] *= 0.75
            series[day, IDX["daily_activity"]] *= 0.85
        # 约束合理范围
        for name in ("sad_ratio", "sleep_efficiency", "deep_sleep_ratio"):
            series[day, IDX[name]] = np.clip(series[day, IDX[name]], 0.0, 1.0)
        for name in ("distress_events", "avg_speed", "pitch_variability",
                     "sfi", "hrv_rmssd", "daily_activity", "social_turns"):
            series[day, IDX[name]] = max(0.0, series[day, IDX[name]])
    return series


def apply_injection(series: np.ndarray, spec: dict, seed: int) -> np.ndarray:
    """按 spec 在指定天段注入异常/混淆。spec:
        {type: block|drift|spike, start, end, features:{name:target}}
    """
    rng = np.random.RandomState(seed + 1)
    s = series.copy()
    typ = spec["type"]
    start, end = spec["start"] - 1, spec["end"] - 1  # 1-based → 0-based
    feats = spec["features"]
    for day in range(start, end + 1):
        for name, target in feats.items():
            i = IDX[name]
            if typ == "block":      # 恒定偏移到 target 附近
                s[day, i] = rng.normal(target, abs(target) * 0.15 + 1e-6)
            elif typ == "drift":    # 从正常线性漂到 target
                frac = (day - start + 1) / (end - start + 1)
                base = BASELINE[name][0]
                s[day, i] = base + (target - base) * frac + rng.normal(0, BASELINE[name][1])
            elif typ == "spike":    # 只这几天尖峰
                s[day, i] = rng.normal(target, abs(target) * 0.1 + 1e-6)
        # 约束
        for name in ("sad_ratio", "sleep_efficiency", "deep_sleep_ratio"):
            s[day, IDX[name]] = np.clip(s[day, IDX[name]], 0.0, 1.0)
    return s

# 场景定义。expect_active/expect_silent 是"应/不应激活的风险类型"。
# expect_max_level：该场景允许达到的最高风险等级（TN 应为 0）。
SCENARIOS = [
    # ---------- 真阳性 TP：应检出、判对类型 ----------
    {
        "id": "TP_depression", "kind": "TP",
        "desc": "典型抑郁：4声学特征齐恶化，连续7天",
        "inject": {"type": "block", "start": 40, "end": 46,
                   "features": {"sad_ratio": 0.20, "avg_speed": 2.5,
                                "pitch_variability": 12.0, "distress_events": 3.0}},
        "expect_active": ["depression"], "expect_silent": [],
        "expect_detect": True,
    },
    {
        "id": "TP_depression_partial", "kind": "TP",
        "desc": "部分抑郁：只 sad_ratio↑ + avg_speed↓（考验方向匹配鲁棒性）",
        "inject": {"type": "block", "start": 40, "end": 46,
                   "features": {"sad_ratio": 0.18, "avg_speed": 2.8}},
        "expect_active": ["depression"], "expect_silent": [],
        "expect_detect": True,
    },
    {
        "id": "TP_sleep", "kind": "TP",
        "desc": "睡眠问题：4睡眠特征恶化，情绪正常（应判睡眠、不该判抑郁）",
        "inject": {"type": "block", "start": 40, "end": 46,
                   "features": {"sleep_efficiency": 0.65, "deep_sleep_ratio": 0.15,
                                "sfi": 14.0, "hrv_rmssd": 28.0}},
        "expect_active": ["sleep_problem"], "expect_silent": ["depression"],
        "expect_detect": True,
    },
    {
        "id": "TP_depression_drift", "kind": "TP",
        "desc": "渐变抑郁：特征缓慢漂移 10 天（考验能否捕捉趋势）",
        "inject": {"type": "drift", "start": 38, "end": 48,
                   "features": {"sad_ratio": 0.22, "avg_speed": 2.3,
                                "pitch_variability": 10.0, "distress_events": 3.5}},
        "expect_active": ["depression"], "expect_silent": [],
        "expect_detect": True,
    },
    # ---------- 真阴性/混淆项 TN：不该报 ----------
    {
        "id": "TN_all_normal", "kind": "TN",
        "desc": "全程正常（带AR1噪声+周节律）：测基础误报率",
        "inject": None,
        "expect_active": [], "expect_silent": ["depression", "sleep_problem", "social_isolation"],
        "expect_detect": False, "expect_max_level": 1,
    },
    {
        "id": "TN_single_spike", "kind": "TN",
        "desc": "单日剧烈波动后恢复：不该升级到提醒/严重",
        "inject": {"type": "spike", "start": 43, "end": 43,
                   "features": {"sad_ratio": 0.30, "avg_speed": 2.0,
                                "pitch_variability": 9.0, "distress_events": 4.0}},
        "expect_active": [], "expect_silent": ["depression"],
        "expect_detect": False, "expect_max_level": 1,
    },
    {
        "id": "TN_cold_flu", "kind": "TN",
        "desc": "感冒：睡眠差5天但情绪正常 → 不该报抑郁",
        "inject": {"type": "block", "start": 40, "end": 44,
                   "features": {"sleep_efficiency": 0.70, "sfi": 11.0}},
        "expect_active": [], "expect_silent": ["depression"],
        "expect_detect": False, "expect_max_level": 2,  # 允许睡眠相关波动，但不该报抑郁
    },
    {
        "id": "TN_weekend_quiet", "kind": "TN",
        "desc": "连续安静4天（social↓ 但<5天门槛）→ 不该报社交孤独",
        "inject": {"type": "block", "start": 41, "end": 44,
                   "features": {"social_turns": 12.0, "daily_activity": 3500.0}},
        "expect_active": [], "expect_silent": ["social_isolation"],
        "expect_detect": False, "expect_max_level": 2,
    },
]

def cleanup(velder: str):
    root = get_project_root() / "data"
    for p in ([root / "features" / velder, root / "baselines" / velder]
              + [root / "raw" / s / velder for s in ("sleep", "activity", "social", "acoustic")]):
        shutil.rmtree(p, ignore_errors=True)
    for sub in ("daily_inference", "alerts"):
        d = root / "logs" / sub
        if d.exists():
            for f in d.glob(f"{velder}_*"):
                f.unlink()


def run_scenario(scn: dict, config) -> dict:
    """跑一个场景：造数据→逐日管道→day21训练。返回观测到的最高等级与激活过的类型。"""
    from src.scheduler.daily_job import run_daily_pipeline, load_raw_sensors
    from src.baseline.trainer import train_initial_baseline

    velder = "D_" + scn["id"]
    cleanup(velder)
    root = get_project_root()
    seed = abs(hash(scn["id"])) % 100000

    series = gen_normal_series(seed)
    if scn.get("inject"):
        series = apply_injection(series, scn["inject"], seed)

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    for day in range(1, N_DAYS + 1):
        ds = (start + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        _generate_raw_data(root / "data" / "raw", velder, ds, series[day - 1], {})

    max_level = 0
    active_types = set()
    try:
        for day in range(1, N_DAYS + 1):
            ds = (start + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            res = run_daily_pipeline(velder, ds, raw_data=load_raw_sensors(velder, ds), config=config)
            if day == BUILD_DAYS:
                train_initial_baseline(velder, config)
            risk = res.get("risk_result") or {}
            # 只统计正式运行期（观察期后，day29+）
            if day >= 29:
                max_level = max(max_level, risk.get("risk_level") or 0)
                for rt in risk.get("risk_types", []):
                    active_types.add(rt.get("risk_key"))
    finally:
        cleanup(velder)
    return {"max_level": max_level, "active_types": active_types}


def evaluate(scn: dict, obs: dict) -> tuple[bool, str]:
    """对照期望打分。返回 (是否通过, 原因)。"""
    active = obs["active_types"]
    lvl = obs["max_level"]
    problems = []

    # 应激活的类型必须都激活
    for t in scn.get("expect_active", []):
        if t not in active:
            problems.append(f"应报未报:{t}")
    # 应沉默的类型必须都没激活
    for t in scn.get("expect_silent", []):
        if t in active:
            problems.append(f"误报:{t}")
    # TN 的等级上限
    if scn["kind"] == "TN":
        cap = scn.get("expect_max_level", 1)
        if lvl > cap:
            problems.append(f"等级越限:L{lvl}>{cap}")
    # TP 必须检出（升到 ≥2）
    if scn.get("expect_detect") and lvl < 2:
        problems.append(f"未检出:最高仅L{lvl}")

    return (len(problems) == 0, "; ".join(problems) if problems else "符合预期")

def main():
    parser = argparse.ArgumentParser(description="合成数据判别力验证（范围2）")
    parser.add_argument("--keep", action="store_true", help="（本脚本每场景自清，此项预留）")
    args = parser.parse_args()

    setup_logger(log_level="ERROR")
    config = load_config()

    print("\n" + "=" * 82)
    print("合成数据判别力验证（范围2：有效果吗）  正常天带AR1噪声+周节律")
    print("=" * 82)
    print(f"{'类别':>4} {'场景':>24} {'最高级':>6} {'激活类型':>26} {'判定':>4}")
    print("-" * 82)

    passed, results = 0, []
    tmap = {"depression": "抑郁", "sleep_problem": "睡眠", "social_isolation": "社交"}
    for scn in SCENARIOS:
        obs = run_scenario(scn, config)
        ok, reason = evaluate(scn, obs)
        passed += ok
        results.append((scn, obs, ok, reason))
        types_str = "、".join(tmap.get(t, t) for t in sorted(obs["active_types"])) or "无"
        print(f"{scn['kind']:>4} {scn['id']:>24} {'L'+str(obs['max_level']):>6} "
              f"{types_str:>24} {'[v]' if ok else '[x]':>4}")

    print("-" * 82)
    print(f"通过 {passed}/{len(SCENARIOS)}")
    print("\n逐场景说明：")
    for scn, obs, ok, reason in results:
        mark = "[v]" if ok else "[x]"
        print(f"  {mark} {scn['id']}: {scn['desc']}")
        if not ok:
            print(f"       └─ 问题: {reason}")
    print("=" * 82)
    print("注：范围2用合成数据测'判别力'（层次B）。正常天已加噪声/周节律/混淆项，")
    print("    但仍非真人数据——不证明'真实有效'（层次C需临床标签）。")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
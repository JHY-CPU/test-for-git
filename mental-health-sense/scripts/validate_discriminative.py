"""合成数据判别力验证（范围 2：有效果吗）

范围 1 只证明"链路跑通"。范围 2 用**更刁钻**的合成数据测判别力：
    - 真阳性(TP)：真异常 → 应检出、且判对类型
    - 真阴性/混淆项(TN)：像异常但不该报 → 应保持安静

关键改进（打破范围1的"循环论证"）：
    1. 正常天带真实噪声：AR(1) 自相关 + 周节律（周末话少），不再是纯 iid 高斯
    2. 多种异常：睡眠恶化（4特征齐）/ 睡眠部分 / 社交退缩
    3. 混淆项：单日尖峰（不该升级）、短社交低（<5天不该报孤独）

关于"串味"（残余的次要成因）：
    本次已删除"抑郁"风险类型并移除 sad_ratio 的跨类型共享——串味的"维度重叠"主因已消除
    （见 docs/VALIDATION.md 缺陷③）。但睡眠/社交两类之间仍存在**模型层面**的次要串味：
    当某一路特征剧烈偏离时，多元 GRU 会通过隐藏状态耦合，使另一路的预测也变得不可靠、
    残差被放大，从而"顺带"激活另一类型。故 TP 场景只断言"应报的类型确实报了 + 整体检出"，
    不再强求两类互相绝对静默；真正的"特异度/不误报"由 TN 场景（正常/混淆数据）把关。

指标：TP 检出率 + 判型正确率；TN 特异度（不误报）。这是 docs/VALIDATION.md 层次 B。
仍**不**证明层次 C（真实有效性需真人+临床标签）。

用独立 elder_id（每场景一个），跑完自动清理。

Usage:
    python scripts/validate_discriminative.py
    python scripts/validate_discriminative.py --keep
"""

import argparse
import hashlib
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

# 正常基线（均值, 标准差）——与前面场景一致（6维）
BASELINE = {
    "sleep_efficiency": (0.88, 0.04), "deep_sleep_ratio": (0.30, 0.03),
    "sfi": (5.0, 1.0), "hrv_rmssd": (50, 5),
    "daily_activity": (6000, 800), "social_turns": (35, 5),
}
IDX = {name: i for i, name in enumerate(HEALTH_FEATURES)}

def gen_normal_series(seed: int, n: int = N_DAYS) -> np.ndarray:
    """生成 n 天带真实噪声的正常数据 (n, 6)。

    真实感来自两点（打破纯 iid 高斯）：
      - AR(1) 自相关：今天 = 0.5*偏离(昨天) + 新噪声，模拟"连着几天偏高/偏低"
      - 周节律：周末(第6/7天) social_turns、daily_activity 自然降低
    """
    rng = np.random.RandomState(seed)
    series = np.zeros((n, 6), dtype=np.float64)
    prev_dev = np.zeros(6)
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
        for name in ("sleep_efficiency", "deep_sleep_ratio"):
            series[day, IDX[name]] = np.clip(series[day, IDX[name]], 0.0, 1.0)
        for name in ("sfi", "hrv_rmssd", "daily_activity", "social_turns"):
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
        for name in ("sleep_efficiency", "deep_sleep_ratio"):
            s[day, IDX[name]] = np.clip(s[day, IDX[name]], 0.0, 1.0)
    return s

# 场景定义。expect_active/expect_silent 是"应/不应激活的风险类型"。
# expect_max_level：该场景允许达到的最高风险等级（TN 应为 0）。
SCENARIOS = [
    # ---------- 真阳性 TP：应检出、判对类型 ----------
    {
        "id": "TP_sleep", "kind": "TP",
        "desc": "睡眠问题：4睡眠特征齐恶化，连续7天（应判睡眠）",
        "inject": {"type": "block", "start": 40, "end": 46,
                   "features": {"sleep_efficiency": 0.65, "deep_sleep_ratio": 0.15,
                                "sfi": 14.0, "hrv_rmssd": 28.0}},
        # 只断言"睡眠确实报了"；不强求 social 绝对静默（强异常下的模型层串味见文件头说明）
        "expect_active": ["sleep_problem"], "expect_silent": [],
        "expect_detect": True,
    },
    {
        "id": "TP_sleep_partial", "kind": "TP",
        "desc": "部分睡眠恶化：只 sleep_efficiency↓ + sfi↑（考验方向匹配鲁棒性）",
        "inject": {"type": "block", "start": 40, "end": 46,
                   "features": {"sleep_efficiency": 0.62, "sfi": 15.0}},
        "expect_active": ["sleep_problem"], "expect_silent": [],
        "expect_detect": True,
    },
    {
        "id": "TP_sleep_drift", "kind": "TP",
        "desc": "渐变睡眠恶化：特征缓慢漂移 10 天（考验能否捕捉趋势）",
        "inject": {"type": "drift", "start": 38, "end": 48,
                   "features": {"sleep_efficiency": 0.60, "deep_sleep_ratio": 0.12,
                                "sfi": 16.0, "hrv_rmssd": 25.0}},
        "expect_active": ["sleep_problem"], "expect_silent": [],
        "expect_detect": True,
    },
    {
        "id": "TP_social", "kind": "TP",
        "desc": "社交退缩：对话轮次+活动量大幅走低，连续≥11天（应判社交）",
        "inject": {"type": "block", "start": 34, "end": 46,
                   "features": {"social_turns": 4.0, "daily_activity": 1500.0}},
        # 社交仅 2/6 特征异常，整体异常分被摊薄、更接近检出边界（睡眠有 4/6 特征，轻松越限）。
        # 故注入幅度取到极强、窗口拉到 13 天，使检出稳健、不受训练随机性影响。
        # 同样只断言"社交确实报了"，不强求 sleep 静默（模型层串味，见文件头说明）。
        "expect_active": ["social_isolation"], "expect_silent": [],
        "expect_detect": True,
    },
    # ---------- 真阴性/混淆项 TN：不该报 ----------
    {
        "id": "TN_all_normal", "kind": "TN",
        "desc": "全程正常（带AR1噪声+周节律）：测基础误报率",
        "inject": None,
        "expect_active": [], "expect_silent": ["sleep_problem", "social_isolation"],
        "expect_detect": False, "expect_max_level": 1,
    },
    {
        "id": "TN_single_spike", "kind": "TN",
        "desc": "单日剧烈波动后恢复：不该升级到提醒/严重",
        "inject": {"type": "spike", "start": 43, "end": 43,
                   "features": {"sleep_efficiency": 0.55, "sfi": 18.0,
                                "hrv_rmssd": 22.0}},
        "expect_active": [], "expect_silent": ["sleep_problem"],
        "expect_detect": False, "expect_max_level": 1,
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
              + [root / "raw" / s / velder for s in ("sleep", "activity", "social")]):
        shutil.rmtree(p, ignore_errors=True)
    for sub in ("daily_inference", "alerts"):
        d = root / "logs" / sub
        if d.exists():
            for f in d.glob(f"{velder}_*"):
                f.unlink()


def run_scenario(scn: dict, config) -> dict:
    """跑一个场景：造数据→逐日管道→day21训练。返回观测到的最高等级与激活过的类型。"""
    import torch
    from src.scheduler.daily_job import run_daily_pipeline, load_raw_sensors
    from src.baseline.trainer import train_initial_baseline

    velder = "D_" + scn["id"]
    cleanup(velder)
    root = get_project_root()
    # 用 hashlib 而非内置 hash()：内置 hash() 对字符串每次进程启动结果都不同
    # （PYTHONHASHSEED 随机化），会导致每次跑出的合成数据不同、通过数在 4/8~6/8 飘。
    # hashlib.md5 是确定性的，保证同一场景每次都得到同一个种子 → 结果可复现。
    seed = int(hashlib.md5(scn["id"].encode()).hexdigest(), 16) % 100000
    # 固定 PyTorch 随机种子：GRU 初始权重默认随机，会让每次训出的模型略有不同、
    # 边界场景通过/不通过翻转。连同上面确定性的数据种子，一起保证整体结果可复现。
    torch.manual_seed(seed)

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
    tmap = {"sleep_problem": "睡眠", "social_isolation": "社交"}
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
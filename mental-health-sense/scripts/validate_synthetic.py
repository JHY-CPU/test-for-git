"""合成数据端到端验证（范围 1：跑通验证）

目的：在无真实数据的开发验证阶段，用合成数据验证"统一每日链路能否端到端跑通"——
即：造 60 天数据 → 冷启动训练 → 逐日推理 → 风险判定，全程不崩、状态流转正确。

这是 docs/VALIDATION.md 层次 A（代码正确性）的落地。它证明"链路通、状态对"，
**不**证明"判别力"（那是范围 2，需混淆项）；更不证明"真实有效"（需真人+临床标签）。

时间线（适配 build_days=21）：
    day 1-21   建档期    → 正常数据，攒够 21 天后训练（→14 个样本）
    day 22-28  观察期    → 只记录不报警（验证观察期生效）
    day 29-39  正常运行  → 带噪正常天（测不误报）
    day 40-46  异常注入  → 睡眠恶化特征，应逐级升到 Level 2/3（测检出+分级）
    day 47-60  恢复+正常 → 应降回 Level 0（测不赖着不降）

用独立 elder_id=V001，跑完自动清理，不碰 E001 真实数据。

Usage:
    python scripts/validate_synthetic.py
    python scripts/validate_synthetic.py --keep   # 保留 V001 数据供人工检查
"""

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_simulation_data import generate_daily_vector, _generate_raw_data
from src.utils.io import get_project_root
from src.utils.logger import setup_logger

VELDER = "V001"          # 验证专用 ID，隔离 E001
N_DAYS = 60
START_DATE = "2026-01-01"
BUILD_DAYS = 21          # 与 settings.yaml 的 training.initial.build_days 对齐

# V001 配置：正常基线沿用 E001，异常注入挪到 day40-46（避开观察期 day22-28）
V001_CONFIG = {
    "name": "验证老人V001",
    "baseline": {
        "sleep_efficiency": (0.88, 0.04), "deep_sleep_ratio": (0.30, 0.03),
        "sfi": (5.0, 1.0), "hrv_rmssd": (50, 5),
        "daily_activity": (6000, 800), "social_turns": (35, 5),
    },
    "anomaly": {
        "start_day": 40, "end_day": 46,
        "features": {"sleep_efficiency": 0.65, "deep_sleep_ratio": 0.15,
                     "sfi": 14.0, "hrv_rmssd": 28.0},
    },
    "description": "day40-46 睡眠恶化特征注入（避开建档21天+观察7天）",
}

def cleanup_velder():
    """删除 V001 的全部产物（features / raw / baselines / 推理日志 / 预警）。"""
    root = get_project_root() / "data"
    targets = [
        root / "features" / VELDER,
        root / "baselines" / VELDER,
        root / "raw" / "sleep" / VELDER,
        root / "raw" / "activity" / VELDER,
        root / "raw" / "social" / VELDER,
        root / "realtime" / VELDER,
    ]
    for t in targets:
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)
    # 推理日志与预警按 {elder}_{date} 命名，逐个删
    for sub in ("daily_inference", "alerts"):
        d = root / "logs" / sub
        if d.exists():
            for f in d.glob(f"{VELDER}_*"):
                f.unlink()


def generate_raw_only(root: Path):
    """只生成 60 天 raw 传感器数据（不预写 features.csv，让管道自己聚合）。"""
    raw_dir = root / "data" / "raw"
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    for day in range(1, N_DAYS + 1):
        date_str = (start_dt + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        vec = generate_daily_vector(day, V001_CONFIG, seed=777)
        _generate_raw_data(raw_dir, VELDER, date_str, vec, V001_CONFIG)


def run_timeline(config):
    """逐日跑 run_daily_pipeline；day21 结束后触发冷启动训练。返回每日结果列表。"""
    from src.scheduler.daily_job import run_daily_pipeline, load_raw_sensors
    from src.baseline.trainer import train_initial_baseline

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    rows = []
    trained = False
    for day in range(1, N_DAYS + 1):
        date_str = (start_dt + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        raw = load_raw_sensors(VELDER, date_str)
        res = run_daily_pipeline(VELDER, date_str, raw_data=raw, config=config)

        inf = res.get("inference_result") or {}
        risk = res.get("risk_result") or {}
        rows.append({
            "day": day, "date": date_str,
            "quality": res.get("data_quality"),
            "inf_status": inf.get("status", "-"),
            "score": inf.get("anomaly_score"),
            "threshold": inf.get("dynamic_threshold"),
            "deviation": inf.get("is_deviation"),
            "obs": inf.get("in_observation_period"),
            "risk_level": risk.get("risk_level"),
            "risk_types": [r.get("risk_type") for r in risk.get("risk_types", [])],
        })

        # 建档期结束（攒够 BUILD_DAYS 天）触发一次冷启动训练
        if day == BUILD_DAYS and not trained:
            m, s, st, ew = train_initial_baseline(VELDER, config)
            rows[-1]["_train"] = f"trained: ewma_n={ew.n} (期望14)"
            trained = True
    return rows

def verify(rows):
    """对照预期打分，返回 (通过项, 失败项) 两个列表。"""
    ok, fail = [], []

    def check(cond, name, detail=""):
        (ok if cond else fail).append(f"{name} {detail}".strip())

    # 1. 全程无崩溃：跑满 60 天
    check(len(rows) == N_DAYS, "跑通", f"完成 {len(rows)}/{N_DAYS} 天，无异常中断")

    # 2. 训练成功：day21 训出 14 个样本
    trow = next((r for r in rows if "_train" in r), None)
    check(trow is not None and "ewma_n=14" in trow.get("_train", ""),
          "冷启动训练", trow.get("_train", "未触发") if trow else "未触发")

    # 3. 状态流转：观察期 / success 各阶段出现在正确区间
    obs_days = [r["day"] for r in rows if r["inf_status"] == "observation"]
    succ_days = [r["day"] for r in rows if r["inf_status"] == "success"]
    check(all(22 <= d <= 28 for d in obs_days) and len(obs_days) > 0,
          "观察期", f"观察期落在 day{min(obs_days)}-{max(obs_days)}（期望22-28）" if obs_days else "无观察期")
    check(len(succ_days) > 0 and min(succ_days) >= 29,
          "success流转", f"success 从 day{min(succ_days)} 起" if succ_days else "无 success")

    # 4. 字段完整性：success 天必须带齐关键字段
    bad = [r["day"] for r in rows if r["inf_status"] == "success"
           and (r["score"] is None or r["threshold"] is None or r["risk_level"] is None)]
    check(not bad, "字段完整", "success 天字段齐全" if not bad else f"缺字段: day{bad}")

    # 5. 检出（信息性）：异常期 day40-46 是否升级到 Level>=2
    anom = [r for r in rows if 40 <= r["day"] <= 46]
    max_lvl = max((r["risk_level"] or 0) for r in anom) if anom else 0
    check(max_lvl >= 2, "异常检出", f"异常期最高 Level={max_lvl}（期望≥2）")

    # 6. 恢复（信息性）：day 55-60 回落到 Level 0
    tail = [r for r in rows if 55 <= r["day"] <= 60]
    tail_max = max((r["risk_level"] or 0) for r in tail) if tail else 0
    check(tail_max == 0, "恢复降级", f"尾期最高 Level={tail_max}（期望0）")

    return ok, fail


def print_report(rows, ok, fail):
    print("\n" + "=" * 78)
    print(f"合成数据端到端验证（范围1：跑通）  elder={VELDER}  {N_DAYS}天  build_days={BUILD_DAYS}")
    print("=" * 78)
    # 关键节点抽样打印（每阶段头尾 + 异常期全打）
    show = set(list(range(1, 8)) + [BUILD_DAYS, 22, 28, 29] + list(range(40, 47)) + [55, 60])
    print(f"{'day':>3} {'date':>10} {'质量':>5} {'推理状态':>18} {'分数':>7} {'阈值':>7} {'偏离':>4} {'等级':>4}")
    print("-" * 78)
    for r in rows:
        if r["day"] in show:
            sc = f"{r['score']:.3f}" if r["score"] is not None else "-"
            th = f"{r['threshold']:.3f}" if r["threshold"] is not None else "-"
            dv = "是" if r["deviation"] else ("否" if r["deviation"] is not None else "-")
            lv = r["risk_level"] if r["risk_level"] is not None else "-"
            print(f"{r['day']:>3} {r['date']:>10} {str(r['quality']):>5} "
                  f"{r['inf_status']:>18} {sc:>7} {th:>7} {dv:>4} {str(lv):>4}")
    print("-" * 78)
    print(f"[PASS] 通过 {len(ok)} 项：")
    for x in ok:
        print(f"   [v] {x}")
    if fail:
        print(f"[FAIL] 失败 {len(fail)} 项：")
        for x in fail:
            print(f"   [x] {x}")
    else:
        print("[ALL PASS] 全部通过：链路端到端跑通，状态流转正确。")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="合成数据端到端验证（范围1）")
    parser.add_argument("--keep", action="store_true", help="保留 V001 数据供人工检查")
    args = parser.parse_args()

    setup_logger(log_level="WARNING")  # 压掉 INFO，只看报告
    from src.utils.io import load_config
    config = load_config()

    cleanup_velder()  # 先清残留，保证可复现
    try:
        generate_raw_only(get_project_root())
        rows = run_timeline(config)
        ok, fail = verify(rows)
        print_report(rows, ok, fail)
    finally:
        if not args.keep:
            cleanup_velder()
            print(f"（V001 数据已清理；加 --keep 可保留）")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

"""合成数据端到端验证（范围 1：跑通验证，双轨）

目的：在无真实数据的开发验证阶段，验证"双轨每日链路能否端到端跑通"——
即：造 60 天数据 → 双轨建档 → 逐日推理 → 三类风险判定，全程不崩、状态流转正确。

这是 docs/VALIDATION.md 层次 A（代码正确性）的落地。它证明"链路通、状态对、
两轨互不串扰"，**不**证明"判别力"（那是范围 2，需混淆项）；
更不证明"真实有效"（需真人 + 临床标签 + 量表对照）。

时间线（build_days=35，已取消观察期）：
    day  1-35  建档期    → 正常数据，攒够 35 天后两轨各自训练
    day 36-39  正常运行  → 带噪正常天（测不误报）
    day 40-46  睡眠恶化  → 应升到 Level 2/3，且**社交轨保持安静**
    day 47-49  恢复
    day 50-58  社交退缩  → 应升到 Level 2/3，且**睡眠轨保持安静**
    day 59-60  恢复      → 应降回 Level 0（测不赖着不降）

两段异常错开是为了验证双轨的信号隔离——这是本次重构的核心收益。

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

from src.baseline.scaler_utils import TRACKS
from src.utils.io import get_project_root
from src.utils.logger import setup_logger

VELDER = "V001"          # 验证专用 ID，隔离 E001
N_DAYS = 60
START_DATE = "2026-01-01"
BUILD_DAYS = 35          # 与 settings.yaml 的 training.initial.build_days 对齐

# 异常段（与 generate_simulation_data 保持一致）
SLEEP_ANOMALY = (40, 46)
SOCIAL_ANOMALY = (50, 58)


def cleanup_velder():
    """删除 V001 的全部产物（features / raw / baselines / 推理日志 / 预警）。"""
    root = get_project_root() / "data"
    targets = [
        root / "features" / VELDER,
        root / "baselines" / VELDER,
        root / "raw" / "sleep" / VELDER,
        root / "raw" / "activity" / VELDER,
        root / "raw" / "camera" / VELDER,
    ]
    for t in targets:
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)
    # ★ 清理白名单必须覆盖**所有**会按 elder_id 落盘的日志目录。
    # mpdd_evidence 是 bee653e 才接进 daily_job 的输出，白名单没跟着更新，
    # 于是每跑一次验证就往仓库里灌数百个 V001_/D_* 文件，还被顺手 git add 了进去
    # （实测残留 605 个）。新增落盘目录时必须同步这里。
    # `alert_state` 的文件名是 `{elder_id}.json`（无日期后缀），不匹配
    # `{VELDER}_*`，所以要单独删——漏了它会把验证残留留在仓库里。
    for sub in ("daily_inference", "mpdd_evidence", "depression"):
        d = root / "logs" / sub
        if d.exists():
            for f in d.glob(f"{VELDER}_*"):
                f.unlink()
    state = root / "logs" / "alert_state" / f"{VELDER}.json"
    state.unlink(missing_ok=True)


def generate_raw_only(root: Path):
    """
    只生成 60 天 raw 传感器数据，不预写 features CSV——让管道自己聚合。

    这样验证的是"聚合→填充→校验→推理"整条链，而不是跳过前半段。
    """
    import scripts.generate_simulation_data as gen

    raw_dir = root / "data" / "raw"
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    seed = 777

    # GRU 初始权重也必须固定，否则每次跑的通过项会变。
    # 数据种子（777）只固定了特征序列，两轨 GRU 的随机初始化不受它影响；
    # 建档期只有 35 天样本、模型又刻意取小，初值差异足以让"擦线"的场景
    # 在两次运行间翻转（实测 social_decline 激活与否会飘）。
    # 与 validate_discriminative.py 同一处理。
    import torch
    torch.manual_seed(seed)

    original_elder = gen.ELDER_ID
    gen.ELDER_ID = VELDER   # 让 _write_raw 落到 V001 目录
    try:
        for day in range(1, N_DAYS + 1):
            date_dt = start_dt + timedelta(days=day - 1)
            day_key = date_dt.strftime("%Y-%m-%d")
            is_weekend = date_dt.weekday() >= 5

            sleep_vec = gen.generate_sleep_vector(day, seed)
            social_vec, hourly = gen.generate_social_vector(day, seed, is_weekend)
            gen._write_raw(raw_dir, day_key, sleep_vec, social_vec, hourly)
    finally:
        gen.ELDER_ID = original_elder


def run_timeline(config):
    """逐日跑 run_daily_pipeline；建档期结束后触发双轨冷启动训练。"""
    from src.baseline.trainer import train_all_tracks
    from src.scheduler.daily_job import load_raw_sensors, run_daily_pipeline

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    rows = []
    trained = False

    for day in range(1, N_DAYS + 1):
        day_key = (start_dt + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        raw = load_raw_sensors(VELDER, day_key)
        res = run_daily_pipeline(VELDER, day_key, raw_data=raw, config=config)

        inf = res.get("inference_result") or {}
        risk = res.get("risk_result") or {}
        row = {
            "day": day,
            "day_key": day_key,
            "quality": res.get("track_quality", {}),
            "risk_level": risk.get("risk_level"),
            "risk_types": [r.get("risk_key") for r in risk.get("risk_types", [])],
        }
        for track in TRACKS:
            tr = inf.get(track) or {}
            row[track] = {
                "status": tr.get("status", "-"),
                "score": tr.get("anomaly_score"),
                "threshold": tr.get("dynamic_threshold"),
                "deviation": tr.get("is_deviation"),
            }
        rows.append(row)

        if day == BUILD_DAYS and not trained:
            results = train_all_tracks(VELDER, config)
            rows[-1]["_train"] = results
            trained = True

    return rows


def verify(rows):
    """对照预期打分，返回 (通过项, 失败项)。"""
    ok, fail = [], []

    def check(cond, name, detail=""):
        (ok if cond else fail).append(f"{name} {detail}".strip())

    def in_range(day, span):
        return span[0] <= day <= span[1]

    # 1. 全程无崩溃
    check(len(rows) == N_DAYS, "跑通", f"完成 {len(rows)}/{N_DAYS} 天，无异常中断")

    # 2. 双轨建档均成功
    trow = next((r for r in rows if "_train" in r), None)
    train_res = trow.get("_train", {}) if trow else {}
    check(
        all(train_res.get(t) == "success" for t in TRACKS),
        "双轨建档", str(train_res) if train_res else "未触发",
    )

    # 3. 状态流转：训练后两轨都应进入 success（已取消观察期）
    for track in TRACKS:
        succ = [r["day"] for r in rows if r[track]["status"] == "success"]
        check(
            bool(succ) and min(succ) >= BUILD_DAYS + 1,
            f"[{track}] success流转",
            f"从 day{min(succ)} 起" if succ else "无 success 天",
        )
        obs = [r["day"] for r in rows if r[track]["status"] == "observation"]
        check(not obs, f"[{track}] 无观察期", f"实际 {len(obs)} 天" if obs else "已取消")

    # 4. 建档前应走兜底而非完全不检测
    fb = [r["day"] for r in rows
          if r["day"] < BUILD_DAYS and r["sleep"]["status"] == "cold_start_fallback"]
    check(len(fb) > 20, "冷启动兜底", f"建档期 {len(fb)} 天走稳健兜底（消除监测盲区）")

    # 4b. ★ 兜底不能只是"跑了"，它得真的能出等级，否则建档期仍是预警盲区。
    # 历史缺陷：judge/rules 的状态白名单漏了 cold_start_fallback，兜底判出的
    # 偏离全被丢掉，risk_level 恒 0。上面那条断言只数 status，测不到这个缺口。
    fb_days = [r for r in rows if r["day"] < BUILD_DAYS
               and r["sleep"]["status"] == "cold_start_fallback"]
    fb_dev = [r for r in fb_days if r["sleep"]["deviation"]]
    fb_leveled = [r for r in fb_dev if (r["risk_level"] or 0) >= 1]
    if fb_dev:
        check(
            len(fb_leveled) == len(fb_dev),
            "兜底能出等级",
            f"建档期 {len(fb_dev)} 个兜底偏离日全部出了等级"
            if len(fb_leveled) == len(fb_dev)
            else f"{len(fb_dev) - len(fb_leveled)}/{len(fb_dev)} 个兜底偏离日 risk_level=0（预警盲区）",
        )
    else:
        # 本场景建档期本就正常，没有偏离日可断言 —— 那就反过来守"正常天不误报"：
        # 兜底的分是稳健 z（阈值 = fallback_sigma=3.0，正常天可达 1.77），
        # 幅度门槛若用绝对常数会让建档期天天误报 L1。
        noisy = [r["day"] for r in fb_days if (r["risk_level"] or 0) >= 1]
        check(not noisy, "兜底不误报", f"建档期 {len(fb_days)} 天无一误报"
              if not noisy else f"误报 {len(noisy)} 天: day{noisy[:5]}")

    # 5. 字段完整性
    bad = [
        r["day"] for r in rows for t in TRACKS
        if r[t]["status"] == "success"
        and (r[t]["score"] is None or r[t]["threshold"] is None)
    ]
    check(not bad, "字段完整", "success 天字段齐全" if not bad else f"缺字段: day{bad}")

    # 6. 睡眠异常检出
    sleep_hits = [
        r["day"] for r in rows
        if in_range(r["day"], SLEEP_ANOMALY) and r["sleep"]["deviation"]
    ]
    check(len(sleep_hits) >= 6, "睡眠异常检出", f"{len(sleep_hits)}/7 天检出")

    # 7. ★ 信号隔离：睡眠异常期社交轨应基本安静
    social_noise = [
        r["day"] for r in rows
        if SLEEP_ANOMALY[0] + 1 <= r["day"] <= SLEEP_ANOMALY[1]
        and r["social"]["deviation"]
    ]
    check(len(social_noise) <= 1, "睡眠期社交轨隔离",
          f"社交轨误报 {len(social_noise)} 天（期望≤1）")

    # 8. 社交异常检出
    social_hits = [
        r["day"] for r in rows
        if in_range(r["day"], SOCIAL_ANOMALY) and r["social"]["deviation"]
    ]
    check(len(social_hits) >= 8, "社交异常检出", f"{len(social_hits)}/9 天检出")

    # 9. ★ 信号隔离：社交异常期睡眠轨应基本安静
    # 容许 ≤1 天，与第 7 项的睡眠期检查同口径。为什么不要求严格为 0：
    # 正常天带 AR(1) 噪声，单轨单日擦线越阈本就会偶发（这里 day54 是
    # score 1.45 / 阈值 1.29，前后各天 0.67~1.17，无趋势——是噪声不是串味）。
    # 要求恒为 0 等于要求零误报率，那只能靠抬高阈值换取，代价是真异常也漏。
    # 真正的串味会表现为连续多天同向抬升，用"≤1 天"能区分开。
    sleep_noise = [
        r["day"] for r in rows
        if in_range(r["day"], SOCIAL_ANOMALY) and r["sleep"]["deviation"]
    ]
    check(len(sleep_noise) <= 1, "社交期睡眠轨隔离",
          f"睡眠轨误报 {len(sleep_noise)} 天 day{sleep_noise}（期望≤1）"
          if sleep_noise else "睡眠轨全程安静")

    # 10. 风险类型对号入座
    sleep_types = {t for r in rows if in_range(r["day"], SLEEP_ANOMALY) for t in r["risk_types"]}
    social_types = {t for r in rows if in_range(r["day"], SOCIAL_ANOMALY) for t in r["risk_types"]}
    check("sleep_stability" in sleep_types, "睡眠类型激活", str(sorted(sleep_types)))
    check("social_decline" in social_types, "社会类型激活", str(sorted(social_types)))
    check("social_decline" not in sleep_types, "睡眠期无社会类型误激活")

    # 11. 分级：异常期升到 Level≥2
    for name, span in (("睡眠", SLEEP_ANOMALY), ("社交", SOCIAL_ANOMALY)):
        lvls = [r["risk_level"] or 0 for r in rows if in_range(r["day"], span)]
        check(max(lvls, default=0) >= 2, f"{name}期分级", f"最高 Level={max(lvls, default=0)}")

    # 12. 恢复降级
    tail = [r["risk_level"] or 0 for r in rows if r["day"] >= 59]
    check(max(tail, default=0) <= 1, "恢复降级", f"尾期最高 Level={max(tail, default=0)}（期望≤1）")

    # 13. 正常期不误报（day 36-39）
    normal_fp = [
        r["day"] for r in rows if 36 <= r["day"] <= 39 and (r["risk_level"] or 0) >= 2
    ]
    check(not normal_fp, "正常期不误报",
          f"day{normal_fp} 误升到 Level≥2" if normal_fp else "day36-39 无 Level≥2")

    return ok, fail


def print_report(rows, ok, fail):
    print("\n" + "=" * 96)
    print(f"合成数据端到端验证（范围1：跑通，双轨）  elder={VELDER}  "
          f"{N_DAYS}天  build_days={BUILD_DAYS}")
    print("=" * 96)

    show = set(
        list(range(1, 5)) + [BUILD_DAYS - 1, BUILD_DAYS, BUILD_DAYS + 1]
        + list(range(38, 48)) + list(range(50, 60)) + [60]
    )
    print(f"{'day':>3} {'date':>11} | {'睡眠分':>7} {'阈值':>6} {'偏':>2} "
          f"| {'社交分':>7} {'阈值':>6} {'偏':>2} | {'级':>2} 风险类型")
    print("-" * 96)
    for r in rows:
        if r["day"] not in show:
            continue

        def fmt(track):
            tr = r[track]
            sc = f"{tr['score']:.3f}" if tr["score"] is not None else "-"
            th = f"{tr['threshold']:.3f}" if tr["threshold"] is not None else "-"
            dv = "Y" if tr["deviation"] else ("." if tr["deviation"] is not None else "-")
            return f"{sc:>7} {th:>6} {dv:>2}"

        lv = r["risk_level"] if r["risk_level"] is not None else "-"
        types = ",".join(r["risk_types"]) if r["risk_types"] else ""
        print(f"{r['day']:>3} {r['day_key']:>11} | {fmt('sleep')} | {fmt('social')} "
              f"| {str(lv):>2} {types}")

    print("-" * 96)
    print(f"[PASS] 通过 {len(ok)} 项：")
    for x in ok:
        print(f"   [v] {x}")
    if fail:
        print(f"[FAIL] 失败 {len(fail)} 项：")
        for x in fail:
            print(f"   [x] {x}")
    else:
        print("[ALL PASS] 链路端到端跑通，状态流转正确，双轨信号隔离成立。")
    print("=" * 96)
    print("注意：本脚本只验证层次 A（代码正确性）。判别力见 validate_discriminative.py；")
    print("      真实有效性需真人数据 + 量表对照，本仓库无法自证。")


def main():
    parser = argparse.ArgumentParser(description="合成数据端到端验证（范围1，双轨）")
    parser.add_argument("--keep", action="store_true", help="保留 V001 数据供人工检查")
    args = parser.parse_args()

    setup_logger(log_level="WARNING")   # 压掉 INFO，只看报告
    from src.utils.io import load_config
    config = load_config()

    cleanup_velder()   # 先清残留，保证可复现
    try:
        generate_raw_only(get_project_root())
        rows = run_timeline(config)
        ok, fail = verify(rows)
        print_report(rows, ok, fail)
    finally:
        if not args.keep:
            cleanup_velder()
            print("（V001 数据已清理；加 --keep 可保留）")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

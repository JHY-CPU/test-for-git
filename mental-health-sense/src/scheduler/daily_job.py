"""
每日定时任务：常态轨（单人系统，双轨）

每日凌晨 03:00 对被监测的老人执行：
    1. 聚合昨日传感器数据 → 双轨特征向量
    2. 按轨做缺失值处理 + 数据校验
    3. 按轨保存特征到各自的 CSV
    4. 双轨推理（GRU预测 → signed/abs 双残差 → EWMA更新）
    5. 风险判定（三类，含跨轨的作息节律紊乱）
    6. 如果偏离，记录预警

调度时刻 02:00 → 03:00：小贝壳的睡眠报告窗口延伸到当日 22:59 之后才最终成型，
02:00 拉取可能取到未闭合的报告。

★ 按轨独立降级：小贝壳掉线时睡眠轨标 insufficient，社交轨照常出结果。
这是双轨架构的核心收益，实现上体现为两轨各自走完整的
聚合→填充→校验→保存→推理链条，任一环失败只影响本轨。
"""

from datetime import datetime, timedelta

import numpy as np

from src.baseline.scaler_utils import TRACKS, get_feature_names
from src.data_pipeline.aggregator import (
    DataInsufficientError,
    aggregate_sleep_features,
    aggregate_social_features,
    aggregate_track_features,
)
from src.data_pipeline.imputer import impute_missing
from src.data_pipeline.validator import (
    QUALITY_INSUFFICIENT,
    check_prolonged_degradation,
    describe_track_capability,
    escalate_if_offline,
    is_usable_for_inference,
    validate_daily_data,
)
from src.utils.io import DAY_KEY_COL, load_features_csv, save_daily_features
from src.utils.logger import get_logger
from src.utils.status import (
    STATUS_COLD_START,
    STATUS_COLD_START_FALLBACK,
    STATUS_SUCCESS,
    is_evaluable,
)

logger = get_logger(__name__)


def run_daily_pipeline(
    elder_id: str,
    day_key: str | None = None,
    raw_data: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    执行单日全流程：按轨聚合 → 填充 → 校验 → 保存 → 双轨推理 → 判定。

    Args:
        elder_id: 老人ID
        day_key: 自然日（默认昨天）
        raw_data: 原始传感器数据字典 {"sleep":..., "activity":..., "camera":...}
                  None 表示自动从 data/raw/ 读取
        config: 全局配置

    Returns:
        {
            "elder_id": str,
            "day_key": str,
            "track_quality": {"sleep": str, "social": str},
            "inference_result": dict | None,
            "risk_result": dict | None,
            "status": str,
        }
    """
    if day_key is None:
        day_key = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"=== 每日管道启动: {elder_id} @ {day_key} ===")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    if raw_data is None:
        raw_data = load_raw_sensors(elder_id, day_key)

    # 1-4. 每轨独立走完聚合→填充→校验→保存
    track_feature_values = {
        "sleep": aggregate_sleep_features(raw_data.get("sleep")),
        "social": aggregate_social_features(
            raw_data.get("activity"), raw_data.get("camera")
        ),
    }

    track_quality: dict[str, str] = {}
    for track in TRACKS:
        track_quality[track] = _process_track(
            elder_id, day_key, track, track_feature_values[track], config
        )

    logger.info(f"  └─ 数据质量: {track_quality}")

    # 5. 双轨推理（至少一轨可用才跑）
    inference_result = None
    risk_result = None
    alert_result = None
    failure: str | None = None
    usable_tracks = tuple(t for t in TRACKS if is_usable_for_inference(track_quality[t]))

    if usable_tracks:
        try:
            from src.baseline.inference import daily_inference
            inference_result = daily_inference(
                elder_id, day_key, config, tracks=usable_tracks,
                track_quality=track_quality,
            )

            # 某轨 GRU 基线尚未就绪（冷启动期）：用稳健滑动基线兜底，消除建档期盲区
            _apply_cold_start_fallbacks(
                elder_id, day_key, inference_result, usable_tracks, config,
                track_quality=track_quality,
            )

            if is_evaluable(inference_result.get("status")):
                # 6. 风险判定
                from src.risk.judge import quick_judge
                risk_result = quick_judge(elder_id, day_key, config)

                # 7. 预警（按事件去重）。
                #
                # 无条件调用而不是只在 risk_level>=1 时调：回到 L0 也是状态变化，
                # trigger_alert 要靠它把活跃事件关掉并发"缓解"通知。旧写法在
                # L0 那天直接跳过，事件永远停在最后一次 L2/L3 上。
                #
                # ★ 必须传 day_key：补算历史日时不传会把"今天"的日期盖到一条
                #   历史判定上，冷却窗口与事件边界全算错。
                from src.risk.alert import trigger_alert
                alert_result = trigger_alert(
                    elder_id=elder_id,
                    risk_level=risk_result.get("risk_level", 0),
                    risk_types=risk_result.get("risk_types", []),
                    config=config,
                    day_key=day_key,
                )

                # 8. 产出给 MPDD-AVP 的单向证据契约。
                # 此前 build_mpdd_evidence 定义了、单测覆盖了，但没有任何链路
                # 真正产出过它——契约文档描述的交付物在磁盘上并不存在。
                _emit_mpdd_evidence(elder_id, day_key, inference_result, risk_result)

        except Exception as e:
            # ★ 失败必须反映到 status 上。
            #
            # 旧写法把异常吞掉后仍返回 status="success"（那个值只由数据质量决定，
            # 完全不反映推理/判定是否真的跑过）。实测后果：hidden_dim 改过但没重训
            # → load_state_dict 抛 RuntimeError → 整块被吞 → daily_inference 还没
            # 走到 save_daily_result，**当天日志根本不存在** → 脚本照样打印
            # "管道状态: success" 并 exit 0 → cron 与监控全绿。老人当天零监测，
            # 而且这个缺日还会持续侵蚀后面几天的持续性窗口（max_skip_days 一到
            # 就把偏离段切开）。
            failure = f"{type(e).__name__}: {e}"
            logger.error(f"  └─ 推理/判定失败: {e}", exc_info=True)
    else:
        logger.warning("  └─ 两轨均不可用，跳过推理")

    # 长期降级检查：连续 5 天非 valid 应升级为运维告警，
    # 而不是让界面继续显示"一切正常"——长期降级和长期正常必须可区分。
    for track in TRACKS:
        history = _get_recent_quality(elder_id, track, n_days=5)
        if check_prolonged_degradation(history, threshold=5):
            logger.error(
                f"  └─ [{track}] 连续 5 天数据质量不佳，系统已失去该轨监测能力，"
                f"需运维介入并向家属明示『当前数据不足』"
            )

    # ★ status 必须反映"今天到底有没有产出判定"，不能只看数据质量与异常。
    #
    #   旧写法有三个分支：failure / 无可用轨 / success。缺的是第四种情形——
    #   **两轨数据质量都正常，但都没能出分**。它真实存在：输入窗口缺日时
    #   infer_track 返回 data_insufficient，而 track_quality 是 valid
    #   （今天的数据是好的），于是 usable_tracks 非空、failure 为 None
    #   → 判 success → run_daily_pipeline.py 的非零退出码防线也不触发。
    #
    #   实测（删掉一天的特征行后）：
    #       2026-08-16  管道status=success  error=None  risk_result=无(未判定)
    #       2026-08-17  管道status=success  error=None  risk_result=无(未判定)
    #       2026-08-20  管道status=success  error=None  risk_result=无(未判定)
    #   cron 全绿、监控全绿，而这几天连推理日志里都没有可用的分。
    #
    #   这与 TODO.md 已修的"异常被吞后 status 仍 success"是同一条失效链的
    #   第二个入口：上次修的是异常路径，这次是"没抛异常但也没结论"的路径。
    #   判据落在 risk_result 上——它是这条链路的**最终产物**，
    #   有它才说明聚合→推理→判定整条链真的走完了。
    if failure is not None:
        status = "inference_failed"
    elif not usable_tracks:
        status = "skipped_inference"
    elif risk_result is None:
        status = "no_verdict"
        logger.error(
            f"  └─ 两轨数据质量正常（{track_quality}）但均未产出可评估结果，"
            f"当日无风险判定。轨状态: "
            f"{(inference_result or {}).get('track_statuses')}；"
            f"这不是'一切正常'，需运维介入"
        )
    else:
        status = "success"

    log = logger.error if status != "success" else logger.info
    log(f"=== 每日管道完成: {elder_id}, status={status} ===")

    return {
        "elder_id": elder_id,
        "day_key": day_key,
        "track_quality": track_quality,
        "inference_result": inference_result,
        "risk_result": risk_result,
        # 预警结果必须回传：否则"今天被事件去重抑制了"这个事实无处可查、
        # 无处可测。旧写法直接丢弃 trigger_alert 的返回值。
        "alert_result": alert_result,
        "status": status,
        "error": failure,
    }


def _emit_mpdd_evidence(
    elder_id: str,
    day_key: str,
    inference_result: dict,
    risk_result: dict,
) -> None:
    """
    把当日的双轨证据落到 data/logs/mpdd_evidence/{elder}_{day}.json。

    单向契约：GRU → MPDD-AVP，本系统不消费对方任何输出（防止基线被抑郁判定
    反向污染，与 weekly_retrain 只用正常天微调是同一条防污染原则）。

    产出失败不得影响主链路：证据契约是给下游的旁路输出，写不出来只该记一条
    错误日志，不能把已经算好的风险判定连累掉。
    """
    try:
        import json

        from src.risk.judge import build_mpdd_evidence
        from src.utils.io import get_log_dir

        evidence = build_mpdd_evidence(
            elder_id, day_key, inference_result, risk_result
        )
        out_dir = get_log_dir("mpdd_evidence")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{elder_id}_{day_key}.json", "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"  └─ MPDD 证据契约产出失败（不影响主链路）: {e}")


def _process_track(
    elder_id: str,
    day_key: str,
    track: str,
    feature_values: dict,
    config: dict,
) -> str:
    """
    单轨的聚合→填充→校验→保存。返回该轨的 data_quality。

    聚合失败（缺 ≥3 维）时仍写一行全 NaN 的记录并标 insufficient——
    留下痕迹比什么都不写好，否则特征表会出现无法解释的日期空洞。
    """
    names = get_feature_names(track)

    try:
        feature_vec = aggregate_track_features(track, feature_values)
    except DataInsufficientError as e:
        logger.warning(f"  └─ [{track}] 数据不足: {e}")
        # 写 NaN 而不是 0：这一行的意思是"这天该轨没测到"，不是"各项指标都是 0"。
        # 旧写法 np.nan_to_num(placeholder, nan=0.0) 会把一整行 0 落进特征表，
        # 而 0 在原始量纲下是极端离群值（实测 night_hr_mean=0 → z=−15.2）。
        # 虽然这行被标了 insufficient、不进训练也不进推理，但它仍然是一条
        # 事实错误的记录，排查时会误导人。
        #
        # ★ 离线升级必须在**这条**分支上也做一次。
        #   聚合阶段抛异常时整个 validate_daily_data 都被跳过，而离线判据原本只
        #   长在它里面——于是 QUALITY_OFFLINE 在生产链路里不可达（实测连跑 7 天
        #   全缺仍只记 insufficient）。判据抽到 validator.escalate_if_offline，
        #   两个产出点共用一份，绝不各写一份。
        recent_quality = _get_recent_quality(elder_id, track, before_day_key=day_key)
        quality = escalate_if_offline(QUALITY_INSUFFICIENT, recent_quality)
        placeholder = np.full(len(names), np.nan, dtype=np.float64)
        save_daily_features(
            elder_id, day_key, placeholder, track,
            missing_count=e.missing_count, data_quality=quality,
        )
        return quality

    # 前向填充基准：该轨最近一条 valid 记录
    prev_vec = _get_prev_valid_vector(elder_id, track, day_key)
    filled_vec, missing_count, missing_names = impute_missing(
        feature_vec, track, prev_vec
    )

    if missing_names:
        logger.info(f"  └─ [{track}] 无法填充的特征: {missing_names}")

    # 排除今天自己：重跑时 CSV 里已有首跑写下的今天，算进去会让离线判据翻档
    recent_quality = _get_recent_quality(elder_id, track, before_day_key=day_key)
    quality = validate_daily_data(
        filled_vec, missing_count, track, recent_quality,
        missing_features=missing_names,
    )

    # 关键特征缺失时必须明示能力受限，不能让"测不到"沉默地显示为"一切正常"
    capability_note = describe_track_capability(track, missing_names)
    if capability_note:
        logger.warning(f"  └─ [{track}] {capability_note}")

    # 社交轨的 5 维并非相互独立（RA/IV/activity_counts 同源于小时活动序列），
    # 设备故障是成组失效而非随机掉维。缺维时给出成组诊断，运维才知道该去看哪台设备——
    # 只报"缺了 2 维"没法定位是 C6c 离线还是 T1C 欠压。
    if track == "social" and missing_names:
        from src.data_pipeline.aggregator import diagnose_social_failure

        diagnosis = diagnose_social_failure(feature_values)
        for reason in diagnosis["reasons"]:
            logger.warning(f"  └─ [social] 成组失效诊断: {reason}")

    save_daily_features(
        elder_id, day_key, filled_vec, track,
        missing_count=missing_count, data_quality=quality,
    )
    return quality


def _get_prev_valid_vector(elder_id: str, track: str, day_key: str) -> np.ndarray | None:
    """取该轨在 day_key 之前最近一条 valid 记录的特征向量"""
    try:
        df = load_features_csv(elder_id, track)
    except FileNotFoundError:
        return None

    names = get_feature_names(track)
    prev = df[(df["data_quality"] == "valid") & (df[DAY_KEY_COL] < day_key)]
    prev = prev.sort_values(DAY_KEY_COL).tail(1)
    if len(prev) == 0:
        return None
    return prev[names].to_numpy(dtype=np.float64).flatten()


def _get_recent_quality(
    elder_id: str, track: str, n_days: int = 5, before_day_key: str | None = None
) -> list[str]:
    """获取某轨最近N天的数据质量列表（按 day_key 升序）。

    ★ before_day_key 用于把**今天自己**排除在外。

      不排除的话，同一天重跑会得到不同的持久化质量档：
        首跑：CSV 里还没有今天这行 → recent = [D-3, D-2, D-1] → 判 insufficient
        重跑：CSV 已有首跑写下的今天 → recent = [D-2, D-1, D] → 三天全 insufficient
              → validator 判 **offline**
      同一份输入因为跑了几次而落到不同的质量档，`check_prolonged_degradation`
      的运维告警也跟着变。补算与重跑必须是幂等的（README 明写这一点）。

    ★ 按**自然日**取，缺日补 `QUALITY_INSUFFICIENT`。

      旧实现取"最近 N 条记录"，与日历无关。而"那天压根没跑"（宕机 / cron 漏
      触发）根本不写 CSV 行，于是那些日子在这个列表里**不存在**——
      `validate_daily_data` 的离线判据看的是"末尾 3 条是否全部非 valid"，
      三条来自三个不相邻的日子，`QUALITY_OFFLINE` 在生产链路里因此不可达
      （全仓只有单测构造过它）。设备连着几天没上报，界面上仍是"一切正常"，
      正好破在"长期降级和长期正常必须可区分"这条原则上。

      缺日按 insufficient 计：那天确实没有可信数据，与"跑过但缺 ≥3 维"
      在运维含义上是同一件事——都需要有人去看设备。
    """
    from datetime import datetime, timedelta

    try:
        df = load_features_csv(elder_id, track)
    except FileNotFoundError:
        return []
    if before_day_key is not None:
        df = df[df[DAY_KEY_COL] < before_day_key]
    if len(df) == 0:
        return []

    by_day = dict(zip(df[DAY_KEY_COL].astype(str), df["data_quality"]))

    # 基准日：给了 before_day_key 就以它的前一天为终点，否则以表里最后一天为终点
    try:
        if before_day_key is not None:
            end_dt = datetime.strptime(before_day_key, "%Y-%m-%d") - timedelta(days=1)
        else:
            end_dt = datetime.strptime(str(df[DAY_KEY_COL].max()), "%Y-%m-%d")
    except ValueError:
        # 日期解析不了就退回旧的按条数取法，不因为一个脏 day_key 让整条链路失败
        recent = df.sort_values(DAY_KEY_COL, ascending=False).head(n_days)
        return recent.sort_values(DAY_KEY_COL)["data_quality"].tolist()

    out: list[str] = []
    for offset in range(n_days - 1, -1, -1):
        key = (end_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        out.append(by_day.get(key, QUALITY_INSUFFICIENT))
    return out


def _apply_cold_start_fallbacks(
    elder_id: str,
    day_key: str,
    inference_result: dict,
    tracks: tuple[str, ...],
    config: dict,
    track_quality: dict[str, str] | None = None,
) -> None:
    """
    对处于 cold_start 的轨启用稳健滑动基线兜底，就地改写 inference_result。

    按轨独立：睡眠轨基线已就绪、社交轨还在建档时，只有社交轨走兜底。
    """
    quality_map = track_quality or {}
    changed = False
    for track in tracks:
        track_result = inference_result.get(track)
        if not isinstance(track_result, dict):
            continue
        if track_result.get("status") != STATUS_COLD_START:
            continue

        fb = _cold_start_fallback_track(
            elder_id, day_key, track, config, data_quality=quality_map.get(track)
        )
        if fb is not None:
            inference_result[track] = fb
            changed = True

    if not changed:
        return

    # 兜底改变了各轨结果，整体状态与连续天数要跟着重算并落盘
    statuses = {
        t: inference_result[t].get("status")
        for t in tracks if isinstance(inference_result.get(t), dict)
    }
    inference_result["track_statuses"] = statuses
    if any(is_evaluable(s) for s in statuses.values()):
        inference_result["status"] = STATUS_SUCCESS

    inference_result["is_deviation"] = any(
        isinstance(inference_result.get(t), dict)
        and inference_result[t].get("is_deviation", False)
        for t in tracks
    )

    from src.utils.io import load_daily_results, save_daily_result
    # end_day_key 必须钉在被处理的那一天：不给会拿磁盘上**最新**几天的日志去数
    # 连续偏离天数，补算历史日时窗口整体漂到最近几天（VALIDATION §9 缺陷①
    # "补算历史日判成最新日"的残留入口）。与 inference.daily_inference 里的
    # 同一段统计保持一致。
    recent = load_daily_results(elder_id, n_days=7, end_day_key=day_key)
    consecutive = 0
    for day_result in reversed(recent):
        if day_result.get("day_key") == day_key:
            continue
        if day_result.get("is_deviation", False):
            consecutive += 1
        else:
            break
    inference_result["consecutive_deviation_days"] = consecutive + (
        1 if inference_result["is_deviation"] else 0
    )
    save_daily_result(elder_id, day_key, inference_result)


def _cold_start_fallback_track(
    elder_id: str,
    day_key: str,
    track: str,
    config: dict,
    data_quality: str | None = None,
) -> dict | None:
    """
    某轨的冷启动兜底：GRU 基线就绪前，用中位数/MAD 稳健基线做基础离群检测。

    Returns:
        与 infer_track 结构兼容的结果字典（status="cold_start_fallback"），
        数据不足以兜底时返回 None。

    注意：本函数**整个替换** inference_result[track]，所以 infer_track 写进去的
    字段（尤其是 data_quality）必须在这里原样补上，否则标记会在兜底路径上丢掉。
    """
    cs_cfg = config.get("cold_start", {})
    if not cs_cfg.get("fallback_enabled", True):
        return None

    min_days = cs_cfg.get("fallback_min_days", 5)
    lookback = cs_cfg.get("fallback_lookback", 14)
    sigma = cs_cfg.get("fallback_sigma", 3.0)

    names = get_feature_names(track)

    try:
        df = load_features_csv(elder_id, track)
    except FileNotFoundError:
        return None

    hist_df = df[(df["data_quality"] == "valid") & (df[DAY_KEY_COL] < day_key)]
    hist_df = hist_df.sort_values(DAY_KEY_COL).tail(lookback)

    if len(hist_df) < min_days:
        logger.info(
            f"  └─ [{track}] 冷启动兜底：历史有效数据不足（{len(hist_df)}/{min_days}天），暂不检测"
        )
        return None

    today_df = df[df[DAY_KEY_COL] == day_key]
    if len(today_df) == 0:
        return None

    history = hist_df[names].to_numpy(dtype=np.float64)
    today_vec = today_df[names].to_numpy(dtype=np.float64).flatten()

    from src.baseline.cold_start_fallback import fallback_deviation_check
    from src.utils.io import get_feature_weight_array

    weights = get_feature_weight_array(track)
    fb = fallback_deviation_check(history, today_vec, weights, track, sigma=sigma)

    is_weekend = datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5

    logger.info(
        f"  └─ [{track}] 冷启动兜底: score={fb['anomaly_score']:.4f}, "
        f"threshold={fb['threshold']:.2f}, deviation={fb['is_deviation']} "
        f"(稳健基线 n={len(history)}, 跳过维={fb['skipped_features']})"
    )

    return {
        "track": track,
        "anomaly_score": fb["anomaly_score"],
        "static_threshold": fb["threshold"],
        "ewma_threshold": fb["threshold"],
        "dynamic_threshold": fb["threshold"],
        "is_deviation": fb["is_deviation"],
        "signed_residuals": fb["feature_z"],
        "abs_residuals": fb["feature_z_abs"],
        "signed_z": fb["feature_z"],
        # 兜底期的 z 分来自稳健基线而非 GRU 残差，方向可用（这正是修掉 np.abs 的收益）
        "signed_available": True,
        "ewma_pool": "weekend" if (track == "social" and is_weekend) else "default",
        "data_quality": data_quality,
        "valid_features": fb["valid_features"],
        "skipped_features": fb["skipped_features"],
        "in_observation_period": True,
        "status": STATUS_COLD_START_FALLBACK,
        "method": fb["method"],
    }


def load_raw_sensors(elder_id: str, day_key: str) -> dict:
    """
    从 data/raw/ 读取各路传感器原始数据。

    Returns:
        {"sleep":..., "activity":..., "camera":...}，缺失的路为 None。
    """
    import json
    from pathlib import Path

    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

    def _load_json(subdir: str) -> dict | None:
        filepath = raw_dir / subdir / elder_id / f"{day_key}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    return {
        "sleep": _load_json("sleep"),
        "activity": _load_json("activity"),
        "camera": _load_json("camera"),
    }

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

    if failure is not None:
        status = "inference_failed"
    elif not usable_tracks:
        status = "skipped_inference"
    else:
        status = "success"

    log = logger.error if failure else logger.info
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
        placeholder = np.full(len(names), np.nan, dtype=np.float64)
        save_daily_features(
            elder_id, day_key, placeholder, track,
            missing_count=e.missing_count, data_quality=QUALITY_INSUFFICIENT,
        )
        return QUALITY_INSUFFICIENT

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
    """
    try:
        df = load_features_csv(elder_id, track)
    except FileNotFoundError:
        return []
    if before_day_key is not None:
        df = df[df[DAY_KEY_COL] < before_day_key]
    recent = df.sort_values(DAY_KEY_COL, ascending=False).head(n_days)
    return recent.sort_values(DAY_KEY_COL)["data_quality"].tolist()


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
    recent = load_daily_results(elder_id, n_days=7)
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

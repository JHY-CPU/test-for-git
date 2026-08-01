"""
每日推理引擎（双轨）

每日凌晨对每一轨各执行一遍：
    1. 加载该轨的模型、scaler、残差统计、EWMA 池
    2. 获取最近7天 + 今天的该轨特征向量
    3. 归一化 + GRU预测
    4. 计算 signed 与 abs 双残差 → anomaly_score
    5. 动态阈值判断 → is_deviation
    6. 更新对应的 EWMA 池（社交轨按 is_weekend 选池）
    7. 记录日志

★ 双残差契约（本模块最容易写错的地方）

    signed_residual[i] = observed[i] − predicted[i]      ← 判方向
    abs_residual[i]    = |signed_residual[i]|            ← 打幅度分

    anomaly_score = Σ(abs_residual · w) / Σw
    方向判定       = signed_residual 标准化后与 direction 比对

符号约定是 `observed − predicted`（统计学通行定义），好处是名字与符号一致：
实际值**下降** → signed **为负** → direction="down" 判 `signed_z < −threshold`。
反过来的约定（pred − observed）会让"下降"对应正数，是反直觉且极易写反的。

两套统计量都在 calibration/holdout 段估计（见 trainer.py），不用训练集。
"""

from datetime import datetime, timedelta

import numpy as np
import torch

from src.baseline.ewma import TrackEWMAPools
from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import (
    TRACKS,
    get_feature_names,
    transform_data,
    validate_track,
)
from src.utils.io import (
    get_baseline_dir,
    get_daily_vector,
    get_feature_weight_array,
    get_feature_window,
    get_model_filename,
    get_scaler_path,
    get_track_meta,
    load_daily_results,
    load_residual_stats,
    save_daily_result,
)
from src.utils.logger import get_logger
from src.utils.status import (
    COLD_START_STATUSES,
    STATUS_COLD_START,
    STATUS_DATA_INSUFFICIENT,
    STATUS_OBSERVATION,
    STATUS_SUCCESS,
    is_evaluable,
)

logger = get_logger(__name__)


def _is_weekend(day_key: str) -> bool:
    return datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5


def _normalize_stats(residual_stats: dict, dim: int) -> dict:
    """
    兼容旧格式的残差统计。

    新格式：{"signed": {"mean","std"}, "abs": {"mean","std"}}
    旧格式：{"mean","std"}（abs 残差的统计，无方向信息）

    旧格式下 signed 统计不可得——用 abs 的 std 顶替会造成量纲错配，
    所以这里明确标记 signed 不可用，让上层拒绝做方向判定而不是给出错误结论。
    """
    if "signed" in residual_stats and "abs" in residual_stats:
        return {**residual_stats, "signed_available": True}

    if "mean" in residual_stats and "std" in residual_stats:
        logger.warning(
            "  └─ 检测到旧格式残差统计（仅 abs，无方向信息）。"
            "方向性风险判定将被跳过，请重新建档以获得 signed 统计。"
        )
        abs_stats = {
            "mean": np.asarray(residual_stats["mean"], dtype=np.float64),
            "std": np.asarray(residual_stats["std"], dtype=np.float64),
        }
        return {
            "signed": {"mean": np.zeros(dim), "std": abs_stats["std"]},
            "abs": abs_stats,
            "signed_available": False,
        }

    raise ValueError(f"Unrecognized residual_stats structure: {list(residual_stats)}")


def _load_prior_thresholds(elder_id: str, day_key: str, track: str) -> dict | None:
    """取该日首跑落盘的三个阈值（重跑时复用，保证判定幂等）。

    读不到就返回 None，由调用方退回按当前池计算——旧日志没有这些字段时
    不该因此报错，那只是回到修复前的行为，不会更糟。
    """
    import json

    from src.utils.io import get_log_dir

    path = get_log_dir("daily_inference") / f"{elder_id}_{day_key}.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            prior_track = (json.load(f) or {}).get(track)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(prior_track, dict):
        return None

    # ★ 只复用**同一条产出路径**写下的阈值，量纲不同的绝不能互相顶替。
    #
    #   建档期的日子走 cold_start_fallback，它落盘的三个阈值都等于
    #   `cold_start.fallback_sigma`（稳健加权 |z| 的量纲，默认 3.0）；GRU 轨是
    #   加权归一化残差（正常 0.5~1.0、阈值 1.3~1.6）。
    #
    #   建档完成后补算一个建档期内的日子会同时满足两个条件：基线文件已存在
    #   （走 GRU 路径）、`day_key <= ewma.last_day_key`（已喂过 → 判为重跑），
    #   于是把 3.0 当成 GRU 轨的 dynamic_threshold —— `is_deviation` 恒为 False，
    #   而该字段驱动 consecutive / risk_type_qualifies / 微调排除集 / 周报统计。
    #
    #   与 judge._severity 的"同一个绝对常数对两者不是同一件事"是同一条
    #   量纲不可比；幂等复用也不该跨尺度。
    if prior_track.get("status") in COLD_START_STATUSES:
        return None

    keys = ("static_threshold", "ewma_threshold", "dynamic_threshold")
    if not all(isinstance(prior_track.get(k), (int, float)) for k in keys):
        return None
    return {k: float(prior_track[k]) for k in keys}


def infer_track(
    elder_id: str,
    day_key: str,
    track: str,
    config: dict | None = None,
    data_quality: str | None = None,
) -> dict:
    """
    对某一轨做当日推理。

    Args:
        data_quality: 该轨当日的数据质量（valid/degraded/insufficient/offline）。
            由 daily_job 的校验环节算出后透传进来，**必须随推理结果一起落盘**——
            持续性统计（rules._counts_toward_consecutive）靠它决定某天算不算数。
            此前这个标记只写进 features_{track}.csv、没进推理日志，导致
            "degraded 日既不累加也不打断"这条设计在生产链路里从未生效：
            读不到字段就一律按 valid 计入，传感器抖动照样能攒成预警。

    Returns:
        {
            "track": str,
            "anomaly_score": float,
            "static_threshold": float,
            "ewma_threshold": float,
            "dynamic_threshold": float,
            "is_deviation": bool,
            "signed_residuals": dict,   # 带符号（判方向）
            "abs_residuals": dict,      # 绝对值（打幅度分）
            "signed_z": dict,           # 标准化后的带符号残差
            "signed_available": bool,
            "ewma_pool": str,
            "data_quality": str | None,
            "status": str,
        }
    """
    track = validate_track(track)

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    risk_cfg = config.get("risk", {})
    ewma_cfg = config.get("ewma", {})
    gru_cfg = config.get("gru", {})
    track_gru = gru_cfg.get(track, {})

    sigma = risk_cfg.get("sigma_multiplier", 2.5)
    cold_start_days = risk_cfg.get("cold_start_observation_days", 0)
    window = gru_cfg.get("window", 7)

    names = get_feature_names(track)
    dim = len(names)
    is_weekend = _is_weekend(day_key)

    base = {
        "track": track,
        "anomaly_score": 0.0,
        "static_threshold": 0.0,
        "ewma_threshold": 0.0,
        "dynamic_threshold": 0.0,
        "is_deviation": False,
        "signed_residuals": {},
        "abs_residuals": {},
        "signed_z": {},
        "signed_available": False,
        "ewma_pool": "weekend" if (track == "social" and is_weekend) else "default",
        "data_quality": data_quality,
    }

    # 1. 加载该轨基线文件
    try:
        from src.baseline.scaler_utils import load_scaler
        from src.utils.io import load_gru_model

        scaler = load_scaler(get_scaler_path(elder_id, track))
        model = load_gru_model(
            PersonalBaselineGRU,
            elder_id,
            get_model_filename(track),
            feature_dim=track_gru.get("feature_dim", dim),
            hidden_dim=track_gru.get("hidden_dim", 8),
            num_layers=gru_cfg.get("num_layers", 1),
            dropout=gru_cfg.get("dropout", 0.2),
        )
        residual_stats = _normalize_stats(load_residual_stats(elder_id, track), dim)
    except FileNotFoundError as e:
        logger.info(f"  └─ [{track}] 基线文件缺失，处于冷启动阶段: {e}")
        return {**base, "status": STATUS_COLD_START}

    ewma = TrackEWMAPools.load(
        get_baseline_dir(elder_id), track,
        alpha=ewma_cfg.get("alpha", 0.05),
        max_freeze_days=ewma_cfg.get("max_freeze_days", 14),
    )

    # 2. 获取今日特征与过去 window 天特征
    #
    # ★ 输入窗口按**自然日对齐**取，缺日以全 NaN 行占位（get_feature_window）。
    #
    #   旧实现用 get_feature_vectors + `len(past) < window` 整体放弃。那个判据
    #   数的是"这个区间里磁盘上有几行"，与日历无关，于是**漏跑一天会让之后连续
    #   7 天彻底出不了分**：实测删掉 2026-08-15 一行后，08-16~08-22 全部
    #   data_insufficient、score=0.0，直到那天滑出窗口才恢复。而这几天的
    #   track_quality 两轨都是 valid（今天的数据是好的），daily_job 于是判
    #   status="success"、脚本 exit 0——cron 全绿而老人连续 8 天零监测。
    #
    #   缺日行的 NaN 与"跑过但某维缺测"的 NaN 在下面走**完全同一条**处理路径
    #   （用冻结 scaler 的训练均值补进 GRU 输入、排除出打分），不需要新增分支。
    #   这与 validator 四态里"缺日与降级日走同一条路径"是同一条原则。
    try:
        today_vec = get_daily_vector(elder_id, day_key, track)

        today_dt = datetime.strptime(day_key, "%Y-%m-%d")
        start_dt = today_dt - timedelta(days=window)
        end_dt = today_dt - timedelta(days=1)

        past, missing_days = get_feature_window(
            elder_id,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            track,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  └─ [{track}] 特征数据获取失败: {e}")
        return {**base, "status": STATUS_DATA_INSUFFICIENT, "error": str(e)}

    # ★ 但不能无上限地容忍缺日：窗口里缺得太多，GRU 的输入基本是训练均值，
    #   预测退化成"平均的一天"，残差随之失去意义。上限复用
    #   risk.continuity.max_skip_days（默认 3）——"超过三天没有可信数据，
    #   就不该再假装这是同一段状态"，与 continuity.walk_back_days 的断段语义
    #   保持一致，判级/判型/推理三处对"缺多久算断"给出同一个答案。
    max_gap = risk_cfg.get("continuity", {}).get("max_skip_days", 3)
    if len(missing_days) > max_gap:
        logger.warning(
            f"  └─ [{track}] 输入窗口缺 {len(missing_days)}/{window} 天 "
            f"（上限 {max_gap}）: {missing_days}，无法可靠预测"
        )
        return {
            **base,
            "status": STATUS_DATA_INSUFFICIENT,
            "missing_window_days": missing_days,
        }
    if missing_days:
        logger.info(
            f"  └─ [{track}] 输入窗口缺 {len(missing_days)} 天（以训练均值占位）: "
            f"{missing_days}"
        )

    # ★ 缺测维的处理：填不上的维在 imputer 里保持 NaN（绝不填 0——原始量纲的 0
    #   会变成 −15σ 的假偏离，见 imputer 里的实测数据）。到这里必须做两件事：
    #
    #   1. GRU 输入不能带 NaN（会污染整个前向传播），用**冻结 scaler 的训练均值**
    #      补上。归一化后恰好是 0，也就是原注释想要的"信息中性占位"——只不过
    #      这一次是在正确的空间里做的。
    #   2. 该维**排除出打分**：权重置零后重新归一化。用均值补进去只是让模型能跑，
    #      不代表我们测到了它；把它的残差算进 anomaly_score 等于拿"预测与均值的
    #      差"冒充"观测与预测的差"。
    #
    #   valid_features / skipped_features 是本仓已有的词汇（cold_start_fallback
    #   产出同名字段），判定层与排查都认它。
    missing_mask = np.isnan(today_vec)
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)

    today_filled = np.where(missing_mask, scaler_mean, today_vec)
    past_filled = np.where(np.isnan(past), scaler_mean, past)

    # 3. 归一化
    past_norm = transform_data(scaler, past_filled, track)
    today_norm = transform_data(scaler, today_filled, track)

    # 4. GRU预测
    input_tensor = torch.tensor(
        past_norm.reshape(1, window, dim), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).numpy().flatten()

    # 5. 双残差
    signed_residual = today_norm - pred_norm       # observed − predicted
    abs_residual = np.abs(signed_residual)

    weights = get_feature_weight_array(track)
    effective_weights = np.where(missing_mask, 0.0, weights)
    weight_sum = float(np.sum(effective_weights))
    if weight_sum <= 0:
        # 全维缺失：不可能给出有意义的分。这种情况上游的 aggregator 早该判
        # DataInsufficientError，走到这里说明契约被破坏了，不要静默返回 0 分
        # （0 分会被当成"非常正常的一天"喂进 EWMA 并计入连续统计）。
        logger.warning(f"  └─ [{track}] 全部特征缺测，无法打分")
        return {**base, "status": STATUS_DATA_INSUFFICIENT}

    anomaly_score = float(np.dot(abs_residual, effective_weights) / weight_sum)

    # 6. 阈值：用 abs 统计（与 anomaly_score 同量纲）
    # 权重必须与 anomaly_score 用的**同一套**（含缺测维置零），否则分子分母
    # 不是同一个加权空间，缺一维时分数与阈值会朝不同方向偏。
    abs_mean = np.asarray(residual_stats["abs"]["mean"], dtype=np.float64)
    abs_std = np.asarray(residual_stats["abs"]["std"], dtype=np.float64)
    base_threshold = float(np.dot(abs_mean, effective_weights) / weight_sum)
    std_threshold = float(np.dot(abs_std, effective_weights) / weight_sum)
    static_threshold = base_threshold + sigma * std_threshold

    # 动态EWMA阈值：取 min 保持敏感度，防止老人自然衰退后系统变得不敏感。
    # 如果 EWMA 阈值上升（老人状态变差），仍然用较低的 static_threshold 兜底。
    min_samples = ewma.min_samples_required(config, is_weekend=is_weekend)
    pool_n = ewma.n_samples(is_weekend=is_weekend)
    if pool_n >= min_samples:
        ewma_threshold = ewma.get_threshold(sigma, is_weekend=is_weekend)
        dynamic_threshold = min(static_threshold, ewma_threshold)
    else:
        ewma_threshold = static_threshold
        dynamic_threshold = static_threshold

    # ★ 重跑同一天时复用首跑落盘的阈值。
    #
    # 池里已经含有这一天自己的分（首跑喂进去的、且无法撤回），此刻读到的阈值是
    # "含被判对象的基线"。不复用的话，擦线分数会在重跑时翻转 is_deviation，
    # 而该字段驱动 consecutive / risk_type_qualifies / 微调排除集 / 周报统计。
    # 见 TrackEWMAPools.already_fed 的说明。
    is_rerun = ewma.already_fed(day_key)
    if is_rerun:
        prior = _load_prior_thresholds(elder_id, day_key, track)
        if prior is not None:
            ewma_threshold = prior["ewma_threshold"]
            dynamic_threshold = prior["dynamic_threshold"]
            static_threshold = prior["static_threshold"]
            logger.info(
                f"  └─ [{track}] 重跑该日，复用首跑阈值 "
                f"(dynamic={dynamic_threshold:.4f})，保证判定幂等"
            )

    is_deviation = bool(anomaly_score > dynamic_threshold)

    # 7. 更新对应池并落盘
    # 偏离日冻结（不喂给 EWMA）：否则基线两三天就学会这次异常、阈值追平分数，
    # 持续性异常被自己的历史掩盖，"连续 N 天"永远凑不满。详见 ewma.update 的说明。
    ewma_updated = ewma.update(
        anomaly_score, is_weekend=is_weekend, is_deviation=is_deviation,
        day_key=day_key,
    )
    ewma.save(get_baseline_dir(elder_id))

    # 8. 标准化 signed 残差：只除以 std（尺度、恒正）以保号，不减均值。
    # signed 残差的均值在留出段上估出来接近 0，减掉意义不大；而若误用 abs 的
    # 正均值去减，会把符号整体拉偏，导致 down 方向永远误触发。
    signed_std = np.asarray(residual_stats["signed"]["std"], dtype=np.float64)
    safe_std = np.where(signed_std < 1e-8, 1e-8, signed_std)
    signed_z = signed_residual / safe_std

    # 缺测维不进任何残差字典：它们的"残差"只是"训练均值与预测的差"，
    # 不含今天的观测信息。留在字典里会被 rules 的方向判定当成真实证据读走
    # （_exceeds 读 signed_z），那正是 copresence_min 缺失却满足 down 方向的老路。
    valid_idx = [i for i in range(dim) if not missing_mask[i]]
    signed_residuals = {names[i]: round(float(signed_residual[i]), 4) for i in valid_idx}
    abs_residuals = {names[i]: round(float(abs_residual[i]), 4) for i in valid_idx}
    signed_z_map = {names[i]: round(float(signed_z[i]), 4) for i in valid_idx}

    # 9. 观察期判定：训练后经过的推理次数（不能直接用样本数，训练已预热 EWMA）
    meta = get_track_meta(elder_id, track)
    n_at_train = meta.get("ewma_n_at_train", 0)
    inferences_since_train = ewma.total_samples() - n_at_train
    in_observation = inferences_since_train <= cold_start_days
    status = STATUS_OBSERVATION if in_observation else STATUS_SUCCESS

    return {
        **base,
        "anomaly_score": round(anomaly_score, 4),
        "static_threshold": round(static_threshold, 4),
        "ewma_threshold": round(ewma_threshold, 4),
        # 冻结与去重必须分开报：两者都让 update 返回 False，但含义完全不同。
        # 旧写法 `not ewma_updated` 会让重跑一个正常日也标成 ewma_frozen=true，
        # 而"冻结"在本系统里是有语义的（偏离日不喂基线），排查时会被误导。
        "ewma_frozen": bool(not ewma_updated and not is_rerun),
        "ewma_rerun": is_rerun,
        "dynamic_threshold": round(dynamic_threshold, 4),
        "is_deviation": is_deviation,
        "signed_residuals": signed_residuals,
        "abs_residuals": abs_residuals,
        "signed_z": signed_z_map,
        "signed_available": residual_stats["signed_available"],
        "valid_features": [names[i] for i in valid_idx],
        "skipped_features": [names[i] for i in range(dim) if missing_mask[i]],
        # 输入窗口里有几天是缺日占位的。落盘是为了让"这天的分是在多少缺日之上
        # 算出来的"可查——预测质量随缺日增多而下降，排查时必须看得见。
        "missing_window_days": missing_days,
        # 只有真正喂进去了才 +1：冻结日与重跑日的池样本数不变，
        # 旧写法无条件 +1 会让日志里的 ewma_n 与磁盘上的池对不上。
        "ewma_n": pool_n + (1 if ewma_updated else 0),
        "ewma_min_samples": min_samples,
        "in_observation_period": in_observation,
        "status": status,
    }


def daily_inference(
    elder_id: str,
    day_key: str,
    config: dict | None = None,
    tracks: tuple[str, ...] = TRACKS,
    track_quality: dict[str, str] | None = None,
) -> dict:
    """
    双轨每日推理：每轨独立算分与阈值，互不影响。

    Args:
        track_quality: {轨: data_quality}，由 daily_job 的校验环节算出。
            逐轨写进结果并在顶层留一份镜像——持续性统计要按轨读它，
            界面/排查要能一眼看到当天两轨各自的数据质量。

    Returns:
        {
            "elder_id": str,
            "day_key": str,
            "sleep":  {...},           # infer_track 的返回
            "social": {...},
            "track_quality": {...},
            "status": str,             # 至少一轨 success 即 success
        }
    """
    logger.info(f"每日推理开始: elder_id={elder_id}, day_key={day_key}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    quality_map = track_quality or {}
    result: dict = {"elder_id": elder_id, "day_key": day_key}

    for track in tracks:
        result[track] = infer_track(
            elder_id, day_key, track, config, data_quality=quality_map.get(track)
        )
        tr = result[track]
        logger.info(
            f"  └─ [{track}] score={tr['anomaly_score']:.4f}, "
            f"threshold={tr['dynamic_threshold']:.4f}, "
            f"deviation={tr['is_deviation']}, status={tr['status']}"
        )

    statuses = {t: result[t]["status"] for t in tracks}
    if any(is_evaluable(s) for s in statuses.values()):
        overall = STATUS_SUCCESS
    elif all(s == STATUS_COLD_START for s in statuses.values()):
        overall = STATUS_COLD_START
    else:
        overall = STATUS_DATA_INSUFFICIENT

    result["status"] = overall
    result["track_statuses"] = statuses

    # ★ track_quality 必须覆盖**全部** TRACKS，而不只是本次跑了推理的那几轨。
    #
    # daily_job 只把 is_usable_for_inference 的轨传进 tracks，于是 insufficient /
    # offline 的那一轨既没有 result[track] 子字典、也不在 track_quality 里。
    # 而 rules._counts_toward_consecutive 的三级回退全都落空后会 `return True`，
    # 把这天当成"质量正常但不达标"→ **打断**连续段。
    #
    # 设计要求是"跳过"（既不累加也不打断）。坏掉的正是双轨架构"故障隔离"这条
    # 存在理由本身：小贝壳掉线一天，本该只让睡眠轨那天不计数，实际却把攒了
    # 几天的偏离段清零，规则永远凑不满门槛。
    if quality_map:
        result["track_quality"] = {t: quality_map.get(t) for t in TRACKS}

    # 连续偏离天数：任一轨偏离即算当日偏离（各轨自己的连续天数由 rules.py 分别统计）
    result["is_deviation"] = any(
        result[t].get("is_deviation", False)
        for t in tracks if isinstance(result.get(t), dict)
    )

    recent = load_daily_results(elder_id, n_days=7, end_day_key=day_key)
    consecutive = 0
    for day_result in reversed(recent):
        if day_result.get("day_key") == day_key:
            continue  # 跳过今天自己的旧记录（重算场景）
        if day_result.get("is_deviation", False):
            consecutive += 1
        else:
            break
    result["consecutive_deviation_days"] = consecutive + (1 if result["is_deviation"] else 0)

    save_daily_result(elder_id, day_key, result)
    return result

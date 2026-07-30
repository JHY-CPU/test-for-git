"""
风险等级判定（双轨）

基于连续超标天数 + 风险聚合分数，判定四级风险：
    0 = 正常
    1 = 关注
    2 = 提醒
    3 = 严重

防误报机制：要求连续超标而非单点触发。

双轨后的关键改动：**不跨轨比较绝对分**。两轨的残差尺度不同（睡眠轨 8 维、
社交轨 5 维，权重和也不同），把 anomaly_sleep 和 anomaly_social 放在一起
取最大值或求平均都没有统计意义。等级判定改为"每轨各自算等级，取较高者"。
"""

import numpy as np

from src.baseline.scaler_utils import TRACKS, TRACK_SLEEP, TRACK_SOCIAL
from src.risk.rules import classify_risk_type
from src.utils.io import load_daily_results
from src.utils.logger import get_logger

logger = get_logger(__name__)

RISK_LABELS = {0: "正常", 1: "关注", 2: "提醒", 3: "严重"}


def _track_history(daily_results: list[dict], track: str) -> list[dict]:
    """抽出某轨的历史结果序列（跳过该轨不可用的日子）"""
    out = []
    for day in daily_results:
        tr = day.get(track)
        if isinstance(tr, dict):
            out.append({**tr, "day_key": day.get("day_key") or day.get("date")})
    return out


def _has_adverse_movement(
    track: str,
    track_history: list[dict],
    threshold: float = 1.0,
    any_threshold: float = 2.0,
) -> bool:
    """
    该轨最近的偏离里，是否存在**朝坏方向**的显著变动。

    为什么需要这个判据：anomaly_score 用的是 abs 残差，方向无关——
    "睡眠效率从 0.88 涨到 0.97" 与 "掉到 0.68" 产生同样大的分数。
    只靠分数判等级，会把明显好转判成"严重风险"，家属收到一条
    "您父亲睡眠严重异常"的提醒，而实情是老人睡得更好了。
    这类误报对信任的破坏比漏报更快。

    风险类型规则本身是有方向的（读 signed_z），但类型未激活并不代表
    等级该压住——TP_sleep_minimal 那种"只中两个必选维"就该报却无类型。
    所以这里做的是更宽的判据：只要**有任一维朝坏方向显著动了**就允许升级；
    只有"所有显著变动都朝好方向"时才封顶在 L1。
    """
    from src.utils.io import get_feature_directions, get_feature_names

    try:
        names = get_feature_names(track)
        directions = get_feature_directions(track)
    except Exception:
        # 拿不到方向元数据时不要静默压低等级——宁可保留原等级（可能误报），
        # 也不要因为配置读取失败而把真实风险降级成"关注"。
        return True

    # 闸门只在**有正面证据表明全是好转**时才生效。
    # 数据缺失不是"好转"的证据——判不出方向就不压等级。
    saw_any_z = False
    considered = 0
    adverse_days = 0

    for day in track_history:
        if not day.get("is_deviation", False):
            continue
        if not day.get("signed_available", False):
            return True   # 缺 signed 信息无法判方向 → 不压等级

        considered += 1
        signed_z = day.get("signed_z") or {}
        day_adverse = False
        for name, direction in zip(names, directions):
            z = signed_z.get(name)
            if z is None:
                continue
            saw_any_z = True
            z = float(z)
            if direction == "down" and z <= -threshold:
                day_adverse = True
            if direction == "up" and z >= threshold:
                day_adverse = True
            # direction="any" 的维（如 sleep_onset_clock）本身不含好/坏信息，
            # 门槛要更高：1.3σ 的就寝时间抖动是正常生活波动，
            # 若按 1σ 计入"变坏的证据"，闸门几乎永远放行——
            # 老人睡得全面变好、只是就寝时间挪了一点，就会发"严重风险"。
            # 2σ 以上才当作真正的相位改变。
            if direction == "any" and abs(z) >= any_threshold:
                day_adverse = True

        adverse_days += day_adverse

    # 一个 z 都没读到（signed_z 为空）→ 无从判断，保留原等级
    if not saw_any_z or considered == 0:
        return True

    # 升级本身是"持续偏离"换来的，方向证据也该持续：
    # 要求这段连续偏离里**至少一半**的天数朝坏方向。
    # 否则一周全面好转中夹一天就寝时间抖动（2.26σ）就能放行 L3，
    # 家属收到"严重风险"而老人其实睡得更好了。
    return adverse_days * 2 >= considered


def _judge_single_track(
    track_history: list[dict],
    risk_cfg: dict,
    track: str | None = None,
) -> dict:
    """
    对某一轨独立判等级。

    Returns:
        {"risk_level": int, "consecutive": int, "avg_anomaly": float, "max_anomaly": float}
    """
    consecutive_cfg = risk_cfg.get("consecutive", {})
    thresholds_cfg = risk_cfg.get("anomaly_score_thresholds", {})

    attn_threshold = consecutive_cfg.get("attention", 1)
    warn_threshold = consecutive_cfg.get("warning", 3)
    severe_threshold = consecutive_cfg.get("severe", 5)
    sustained_avg_threshold = thresholds_cfg.get("sustained_avg", 1.0)
    high_spike_threshold = thresholds_cfg.get("high_spike", 1.5)

    usable = [
        d for d in track_history
        if d.get("status") in ("success", "observation")
    ]

    if not usable:
        return {
            "risk_level": 0,
            "consecutive": 0,
            "avg_anomaly": 0.0,
            "max_anomaly": 0.0,
            "evaluable": False,
        }

    recent = usable[-7:]

    # 连续超标天数：从最近往前数，遇到第一个"未偏离"即停
    consecutive = 0
    for day in reversed(recent):
        if day.get("is_deviation", False):
            consecutive += 1
        else:
            break

    scores = [float(d.get("anomaly_score", 0.0)) for d in recent]
    avg_anomaly = float(np.mean(scores))
    max_anomaly = float(np.max(scores))

    # 严重级会触发社区网格员介入 + 强提醒，代价高，必须比"提醒"级更严格，
    # 至少要满足同样的幅度门槛（avg_anomaly > sustained_avg）。否则长达数天、
    # 但每天仅"擦线"越过动态阈值的低幅度偏离，会仅凭连续天数直接升到最高级。
    sustained = avg_anomaly > sustained_avg_threshold
    if consecutive >= severe_threshold and sustained:
        level = 3
    elif consecutive >= warn_threshold and sustained:
        level = 2
    elif consecutive >= attn_threshold or max_anomaly > high_spike_threshold:
        level = 1
    else:
        level = 0

    # 方向闸门：L2/L3 会触发家属提醒/网格员介入，代价高。
    # 若最近的偏离**全部朝好方向**（睡得更好、出门更多），封顶在 L1"关注"——
    # 变化本身值得留意（可能是躁狂期、也可能是数据问题），但不该发风险提醒。
    adverse = True
    if level >= 2 and track is not None:
        # 只看驱动本次升级的那段**连续偏离**（recent 的尾部），不是整个 7 天窗。
        # 用整窗会被窗口里一天过渡期的混合信号翻掉：进入好转的第一两天
        # 预测尚未跟上，个别维会短暂朝坏方向摆，于是闸门永远放行。
        streak = recent[-consecutive:] if consecutive else []
        adverse = _has_adverse_movement(track, streak)
        if not adverse:
            level = 1

    return {
        "risk_level": level,
        "consecutive": consecutive,
        "avg_anomaly": round(avg_anomaly, 4),
        "max_anomaly": round(max_anomaly, 4),
        "adverse_direction": adverse,
        "evaluable": True,
    }


def judge_risk_level(
    elder_id: str,
    daily_results: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """
    判定当前风险等级（双轨各自判级后取较高者）。

    Args:
        elder_id: 老人ID
        daily_results: 近7天双轨推理结果（可选，不提供则自动加载）
        config: 全局配置

    Returns:
        {
            "elder_id": str,
            "risk_level": int,           # 0/1/2/3，两轨取较高
            "risk_label": str,
            "per_track": {"sleep": {...}, "social": {...}},
            "risk_types": list[dict],    # 活跃的风险类型
            "risk_type_qualifies": dict,
            "consecutive_deviation": int,
            "recommendation": str,
        }
    """
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    risk_cfg = config.get("risk", {})

    if daily_results is None:
        daily_results = load_daily_results(elder_id, n_days=7)

    if not daily_results:
        return {
            "elder_id": elder_id,
            "risk_level": 0,
            "risk_label": "正常",
            "per_track": {},
            "risk_types": [],
            "risk_type_qualifies": {},
            "consecutive_deviation": 0,
            "recommendation": "数据不足，无法判定",
        }

    # 1. 每轨独立判级（不跨轨比较绝对分）
    per_track = {}
    for track in TRACKS:
        per_track[track] = _judge_single_track(
            _track_history(daily_results, track), risk_cfg, track=track
        )

    evaluable = [t for t in TRACKS if per_track[t]["evaluable"]]
    risk_level = max((per_track[t]["risk_level"] for t in evaluable), default=0)
    consecutive_deviation = max(
        (per_track[t]["consecutive"] for t in evaluable), default=0
    )

    # 2. 分类风险类型（跨轨规则在 rules.py 内部处理）
    active_risk_types: list[dict] = []
    risk_type_qualifies: dict[str, bool] = {}
    try:
        latest = daily_results[-1]
        risk_type_results = classify_risk_type(
            track_results=latest,
            daily_results=daily_results,
            config=config,
        )
        active_risk_types = [r for r in risk_type_results if r.get("is_active")]
        risk_type_qualifies = {
            r["risk_key"]: bool(r.get("qualifies", False)) for r in risk_type_results
        }
    except Exception as e:
        logger.warning(f"  └─ 风险类型分类跳过: {e}")

    # 3. 生成建议
    recommendation = _generate_recommendation(
        risk_level, active_risk_types, consecutive_deviation, per_track
    )

    logger.info(
        f"风险判定: elder_id={elder_id}, level={risk_level}({RISK_LABELS[risk_level]}), "
        f"sleep={per_track[TRACK_SLEEP]['risk_level']}, "
        f"social={per_track[TRACK_SOCIAL]['risk_level']}, "
        f"consecutive={consecutive_deviation}"
    )

    return {
        "elder_id": elder_id,
        "risk_level": risk_level,
        "risk_label": RISK_LABELS[risk_level],
        "per_track": per_track,
        "risk_types": active_risk_types,
        "risk_type_qualifies": risk_type_qualifies,
        "consecutive_deviation": consecutive_deviation,
        "recommendation": recommendation,
    }


def _generate_recommendation(
    risk_level: int,
    active_risk_types: list[dict],
    consecutive_deviation: int,
    per_track: dict,
) -> str:
    """
    根据风险等级和类型生成处置建议。

    措辞约束：只说行为观察，不下诊断。禁止出现"抑郁症""睡眠障碍""孤独症"。
    """
    if risk_level == 0:
        return "老人状态稳定，无异常检测"

    risk_names = [r["risk_type"] for r in active_risk_types]

    if risk_level == 1:
        if risk_names:
            return f"轻度关注：{'、'.join(risk_names)}指标出现波动，建议持续观察"
        return f"单日轻微偏离（连续{consecutive_deviation}天），建议关注后续变化"

    if risk_level == 2:
        names_str = "、".join(risk_names) if risk_names else "多项指标"
        return (
            f"需要提醒：{names_str}已连续{consecutive_deviation}天偏离个人基线，"
            f"建议子女主动联系老人，了解近期生活状态"
        )

    # risk_level == 3
    if risk_names:
        return (
            f"严重警告：{'、'.join(risk_names)}已连续{consecutive_deviation}天严重偏离基线，"
            f"建议安排上门探访或就医咨询"
        )
    return f"连续{consecutive_deviation}天异常，建议尽快联系老人"


def build_mpdd_evidence(
    elder_id: str,
    day_key: str,
    daily_result: dict,
    risk_result: dict,
) -> dict:
    """
    构建给 MPDD-AVP 的单向证据契约。

    约束（写进契约文档，两侧都要遵守）：
      - MPDD-AVP 可把这些块作为**先验/辅助特征**，但不得让本系统的偏离分
        直接决定抑郁标签。
      - 本系统**不消费** MPDD-AVP 的任何输出，防止基线被抑郁判定反向污染。
        这与 weekly_retrain 只用 is_deviation=False 的正常天微调是同一条防污染原则。
      - signed_z 的符号约定是 observed − predicted：负值 = 低于个人基线。

    Returns:
        契约字典（schema_version 用于两侧对齐）
    """
    sleep = daily_result.get(TRACK_SLEEP, {}) or {}
    social = daily_result.get(TRACK_SOCIAL, {}) or {}
    per_track = risk_result.get("per_track", {})

    def _quality(track_result: dict) -> str:
        status = track_result.get("status")
        if status in ("success", "observation"):
            return "valid"
        if status == "cold_start":
            return "cold_start"
        return "missing"

    sleep_z = sleep.get("signed_z", {}) or {}
    social_z = social.get("signed_z", {}) or {}

    circadian_keys = ("rar_amplitude", "rar_iv")
    circadian_disrupted = any(
        r.get("risk_key") == "circadian_disruption" and r.get("is_active")
        for r in risk_result.get("risk_types", [])
    )

    return {
        "schema_version": "2.1.0",
        "elder_id": elder_id,
        "day_key": day_key,
        "timezone": "Asia/Shanghai",
        "sleep_evidence": {
            "anomaly_score": sleep.get("anomaly_score", 0.0),
            "is_deviation": bool(sleep.get("is_deviation", False)),
            "consecutive_days": per_track.get(TRACK_SLEEP, {}).get("consecutive", 0),
            "signed_z": sleep_z,
            "quality": _quality(sleep),
        },
        "circadian_evidence": {
            "signed_z": {k: social_z[k] for k in circadian_keys if k in social_z},
            "is_disrupted": circadian_disrupted,
            "quality": _quality(social),
        },
        "social_evidence": {
            "anomaly_score": social.get("anomaly_score", 0.0),
            "is_deviation": bool(social.get("is_deviation", False)),
            "consecutive_days": per_track.get(TRACK_SOCIAL, {}).get("consecutive", 0),
            "signed_z": {
                k: v for k, v in social_z.items() if k not in circadian_keys
            },
            "quality": _quality(social),
        },
        "risk_level": risk_result.get("risk_level", 0),
        "note": "单向契约：GRU → MPDD-AVP。本系统不消费 MPDD-AVP 输出。",
    }


def quick_judge(elder_id: str, day_key: str, config: dict | None = None) -> dict:
    """
    快速判定：加载最新推理结果后直接判定。适用于每日调度任务。

    判定后把"今天各风险类型是否达标（qualifies）"写回今天的推理日志，
    使次日的持续性统计能读到今天——这是风险类型能连续累积、最终激活的关键。
    """
    from src.utils.io import save_daily_result

    daily_results = load_daily_results(elder_id, n_days=7)
    result = judge_risk_level(elder_id, daily_results, config)

    # 把今天的 qualifies 写回今天的推理日志（保留 inference 已写入的全部字段）
    qualifies = result.get("risk_type_qualifies")
    if qualifies and daily_results:
        today_log = daily_results[-1]
        log_day = today_log.get("day_key") or today_log.get("date")
        if log_day == day_key:
            today_log["risk_type_qualifies"] = qualifies
            today_log["risk_level"] = result.get("risk_level", 0)
            save_daily_result(elder_id, day_key, today_log)

    return result

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
from src.risk.rules import classify_risk_type, required_history_days
from src.utils.continuity import walk_back_days
from src.utils.io import load_daily_results
from src.utils.logger import get_logger
from src.utils.status import COLD_START_STATUSES, STATUS_COLD_START, is_evaluable

logger = get_logger(__name__)

RISK_LABELS = {0: "正常", 1: "关注", 2: "提醒", 3: "严重"}

# 全链路的 day_key 由 naive `datetime.now()` 推出，所以标称时区必须是**本机时区**，
# 不能写死 Asia/Shanghai——否则容器跑在 UTC 时，契约会用错误的时区标签描述
# 一个 UTC 日期，下游按标签换算就偏一天。
DEFAULT_TIMEZONE = "Asia/Shanghai"


def _local_timezone() -> str:
    """本机时区名；取不到时回落到部署地默认值。"""
    from datetime import datetime

    tz = datetime.now().astimezone().tzinfo
    return str(tz) if tz is not None else DEFAULT_TIMEZONE

# 等级判定只看最近 7 个自然日。与风险类型的持续性统计分开：
# 后者要按各自门槛回溯更远（circadian 降级模式要 7 天达标），
# 但"最近状况有多严重"这个问题只该看最近一周。
LEVEL_WINDOW_DAYS = 7


def _track_history(daily_results: list[dict], track: str) -> list[dict]:
    """抽出某轨的历史结果序列（跳过该轨不可用的日子）"""
    out = []
    for day in daily_results:
        tr = day.get(track)
        if isinstance(tr, dict):
            out.append({**tr, "day_key": day.get("day_key") or day.get("date")})
    return out


def _within_recent_days(
    track_history: list[dict],
    days: int,
    today_key: str | None = None,
) -> list[dict]:
    """
    截到最近 `days` 个自然日（含基准日当天）。

    拿不到日期时退回按条数取尾部——历史日志与单测桩可能没有 day_key，
    这时保持旧行为，不因为缺一个字段就把整段历史丢掉。
    """
    from datetime import datetime, timedelta

    if not track_history:
        return []

    if not today_key:
        today_key = track_history[-1].get("day_key")
    if not today_key:
        return track_history[-days:]

    try:
        cutoff = datetime.strptime(today_key, "%Y-%m-%d") - timedelta(days=days - 1)
    except ValueError:
        return track_history[-days:]

    out = []
    for day in track_history:
        key = day.get("day_key")
        if not key:
            continue
        try:
            if datetime.strptime(key, "%Y-%m-%d") >= cutoff:
                out.append(day)
        except ValueError:
            continue
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


def _severity(day: dict) -> float:
    """
    该日分数相对**它自己那天的动态阈值**的倍数。按定义 severity > 1 ⟺ is_deviation。

    为什么幅度门槛必须用倍数而不是绝对分：两类轨的 anomaly_score 量纲根本不同。
    GRU 轨是加权归一化残差（正常 0.5~1.0，阈值 1.3~1.6）；冷启动兜底轨是稳健
    加权 |z|（阈值就是 fallback_sigma=3.0，正常天能跑到 1.77）。同一个绝对常数
    对两者不是同一件事——1.5 对 GRU 轨是"高峰"，对兜底轨是"再正常不过的一天"。

    倍数是可比的：不管用哪套尺度，"越过自己的阈值多少倍"都表达同一件事。

    无阈值时退回绝对分：历史日志与单测桩不带 dynamic_threshold，退回绝对分
    可保持它们原有的语义，不必为此改造既有数据。
    """
    score = float(day.get("anomaly_score", 0.0))
    threshold = float(day.get("dynamic_threshold") or 0.0)
    return score / threshold if threshold > 0 else score


def _judge_single_track(
    track_history: list[dict],
    risk_cfg: dict,
    track: str | None = None,
    today_key: str | None = None,
) -> dict:
    """
    对某一轨独立判等级。

    Args:
        today_key: 判定基准日。必须由调用方给出而不是取 track_history 的末条——
            该轨今天恰好不可用时，末条是更早的一天，窗口会跟着往前漂。

    Returns:
        {"risk_level": int, "consecutive": int, "avg_anomaly": float,
         "max_anomaly": float, "avg_severity": float, "max_severity": float}
    """
    consecutive_cfg = risk_cfg.get("consecutive", {})
    thresholds_cfg = risk_cfg.get("anomaly_score_thresholds", {})
    # 与 rules 层共用同一个 max_skip_days：一次长时间离线不该把两段无关的偏离
    # 粘成一段，判级与判型对这件事必须给出同样的答案。
    max_skip_days = risk_cfg.get("continuity", {}).get("max_skip_days", 3)

    attn_threshold = consecutive_cfg.get("attention", 1)
    warn_threshold = consecutive_cfg.get("warning", 3)
    severe_threshold = consecutive_cfg.get("severe", 5)
    sustained_severity_threshold = thresholds_cfg.get("sustained_severity", 1.15)
    high_spike_severity_threshold = thresholds_cfg.get("high_spike_severity", 1.5)

    # 等级判定窗口固定为最近 7 个**自然日**。
    # 不能只取"最近 7 条 usable 记录"：判定层现在会加载远多于 7 天的日志
    # （见 rules.required_history_days），按条数取会把窗口悄悄拉宽到几周，
    # "连续 5 天"就可能由散落在两三周里的偏离日凑成。
    #
    # ★ 计入判据必须与 rules 层一致：既要 status 可评估，也要 data_quality 正常。
    #   只看 status 的旧写法漏掉了后者——degraded 日的 status 是 "success"
    #   （is_usable_for_inference 放行 valid+degraded），于是它**既累加也能打断**
    #   consecutive，而 consecutive 正是驱动 L2/L3 的量。见 utils/continuity.py
    #   开头记录的那组误报/漏报场景。
    def _counts(day: dict) -> bool:
        if not is_evaluable(day.get("status")):
            return False
        quality = day.get("data_quality")
        return quality is None or quality == "valid"

    index = {d["day_key"]: d for d in track_history if d.get("day_key")}

    if today_key and index:
        # 按自然日回溯，与风险类型统计同一套走法（含 max_skip_days 打断）
        recent_desc = list(walk_back_days(
            index, today_key,
            max_steps=LEVEL_WINDOW_DAYS - 1, max_skip=max_skip_days,
            counts_fn=_counts, include_today=True,
        ))
        recent = list(reversed(recent_desc))
    else:
        # 拿不到日期时退回按条数取尾部——历史日志与单测桩可能没有 day_key，
        # 这时保持旧行为，不因为缺一个字段就把整段历史丢掉。
        recent = [
            d for d in _within_recent_days(track_history, LEVEL_WINDOW_DAYS, today_key)
            if _counts(d)
        ][-LEVEL_WINDOW_DAYS:]

    if not recent:
        return {
            "risk_level": 0,
            "consecutive": 0,
            "avg_anomaly": 0.0,
            "max_anomaly": 0.0,
            "avg_severity": 0.0,
            "max_severity": 0.0,
            "adverse_direction": True,
            "evaluable": False,
        }

    # 连续超标天数：从最近往前数，遇到第一个"未偏离"即停
    # （不计入统计的日子已在上面被剔除，所以这里数的是"计入的连续日"）
    consecutive = 0
    for day in reversed(recent):
        if day.get("is_deviation", False):
            consecutive += 1
        else:
            break

    # ★ 幅度门槛算在**驱动本次升级的那段连续偏离**上，不是整个 7 天窗。
    # 旧实现用整窗均值，被窗口里的正常天稀释：实测 3 天各 1.45（都实打实越过了
    # 各自的阈值）被前面 4 个正常天拉到 0.907 < 1.0，只报 L1。而方向闸门
    # （下方 _has_adverse_movement）早已改成只看 streak，两个门槛看的不是同一段数据。
    streak = recent[-consecutive:] if consecutive else []
    severities = [_severity(d) for d in recent]
    streak_severities = [_severity(d) for d in streak]

    scores = [float(d.get("anomaly_score", 0.0)) for d in recent]
    streak_scores = [float(d.get("anomaly_score", 0.0)) for d in streak]
    avg_anomaly = float(np.mean(streak_scores)) if streak_scores else 0.0
    max_anomaly = float(np.max(scores))
    avg_severity = float(np.mean(streak_severities)) if streak_severities else 0.0
    max_severity = float(np.max(severities))

    # 严重级会触发社区网格员介入 + 强提醒，代价高，必须比"提醒"级更严格，
    # 至少要满足同样的幅度门槛。否则长达数天、但每天仅"擦线"越过动态阈值的
    # 低幅度偏离，会仅凭连续天数直接升到最高级。
    #
    # 门槛必须 > 1：偏离日按定义 severity > 1，门槛取 1.0 等于恒真，这条防线
    # 会静默失效。1.15 = "整段偏离平均要高出自己阈值 15%，不能只是擦线"。
    sustained = avg_severity > sustained_severity_threshold
    if consecutive >= severe_threshold and sustained:
        level = 3
    elif consecutive >= warn_threshold and sustained:
        level = 2
    elif consecutive >= attn_threshold or max_severity > high_spike_severity_threshold:
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
        "avg_severity": round(avg_severity, 4),
        "max_severity": round(max_severity, 4),
        "adverse_direction": adverse,
        "evaluable": True,
    }


def judge_risk_level(
    elder_id: str,
    daily_results: list[dict] | None = None,
    config: dict | None = None,
    today_key: str | None = None,
) -> dict:
    """
    判定当前风险等级（双轨各自判级后取较高者）。

    Args:
        elder_id: 老人ID
        daily_results: 近7天双轨推理结果（可选，不提供则自动加载）
        config: 全局配置
        today_key: 判定基准日。**补算历史日时必须显式给出**——
            不给就退回按 `daily_results[-1]` 推断，而那是"最新的一条"，
            不是"要判的那一天"。日志覆盖 07-01~08-29 时补算 08-15，
            会拿 08-29 的窗口判 08-15 的等级、按它发预警、写进 08-15 的契约。
            这与 `_judge_single_track` 的 today_key 必须由调用方给出是同一个理由
            （该轨今天恰好不可用时，末条是更早的一天，窗口会跟着往前漂）。

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
        # 加载天数由规则门槛推导，不写死 7——circadian 降级模式要求连续 7 天达标，
        # "今天 + 6 天历史"恰好只有 7 条，历史里有任何一个 degraded 日被跳过，
        # 计数上限就掉到 6，规则在数学上永远不可能激活。详见 required_history_days。
        #
        # end_day_key 截到基准日为止：否则补算历史日时会把该日之后的日志也捞进来，
        # 窗口整体漂到最新那几天。
        daily_results = load_daily_results(
            elder_id, n_days=required_history_days(config), end_day_key=today_key
        )

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
    # 基准日优先用调用方给的；没给才退回末条（历史日志与单测桩没有 day_key 时
    # 也走这条路，保持旧行为）。
    if today_key is None:
        today_key = daily_results[-1].get("day_key") or daily_results[-1].get("date")
    per_track = {}
    for track in TRACKS:
        per_track[track] = _judge_single_track(
            _track_history(daily_results, track), risk_cfg,
            track=track, today_key=today_key,
        )

    evaluable = [t for t in TRACKS if per_track[t]["evaluable"]]
    risk_level = max((per_track[t]["risk_level"] for t in evaluable), default=0)
    consecutive_deviation = max(
        (per_track[t]["consecutive"] for t in evaluable), default=0
    )

    # 2. 分类风险类型（跨轨规则在 rules.py 内部处理）
    #
    # 用基准日那条而不是末条：两者在正常流程下相同，但补算/日志缺失时会分叉，
    # 那时该判的是基准日，不是"手头最新的一条"。
    latest = next(
        (r for r in reversed(daily_results)
         if (r.get("day_key") or r.get("date")) == today_key),
        daily_results[-1],
    )

    active_risk_types: list[dict] = []
    risk_type_qualifies: dict[str, bool] = {}
    try:
        risk_type_results = classify_risk_type(
            track_results=latest,
            daily_results=daily_results,
            config=config,
        )
        active_risk_types = [r for r in risk_type_results if r.get("is_active")]
        risk_type_qualifies = {
            r["risk_key"]: bool(r.get("qualifies", False)) for r in risk_type_results
        }
    except (ValueError, KeyError) as e:
        # ★ 配置/契约类错误必须炸出来，不能吞。
        #
        # 曾经这里是 `except Exception`，后果是：有人把 feature_weights.json 里某个
        # direction 误写成 "decrease" → get_feature_directions 抛 ValueError →
        # 每天被一行 WARNING 吞掉 → 三条规则永远报"无风险类型"，而且因为下方的
        # 写回也跟着不执行，次日所有历史日的 _day_qualifies 都是 False，
        # 持续性计数被永久钉在 1，**任何风险类型在数学上都不可能激活**。
        # 唯一征兆是每天一行 WARNING，而 run_daily_pipeline 照样打印
        # "管道状态: success"。这类静默失效正是本仓反复踩的坑。
        raise
    except Exception as e:
        # 其余异常（例如某条历史日志结构异常）仍然兜住：单日数据问题不该让
        # 整条判定链停摆，但要记 ERROR 而不是 WARNING——它不是正常降级。
        logger.error(f"  └─ 风险类型分类失败（非配置错误，已跳过）: {e}", exc_info=True)

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
        """
        对外呈现的证据强度。cold_start_fallback 走稳健滑动基线而非 GRU，
        证据比 valid 弱但绝不是 missing——报成 missing 会让下游以为"没测到"，
        而实际上那天是有方向、有偏离判定的。
        """
        status = track_result.get("status")
        if is_evaluable(status) and status not in COLD_START_STATUSES:
            return "valid"
        if status in COLD_START_STATUSES or status == STATUS_COLD_START:
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
        # 时区取自配置而非写死：全链路的 day_key 用的是 naive datetime.now()，
        # 容器跑在 UTC 时 day_key 是 UTC 日期却被标成 Asia/Shanghai，
        # 下游按标称时区解读就会错一天。
        "timezone": _local_timezone(),
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
    from src.utils.io import load_config, save_daily_result

    if config is None:
        config = load_config()

    # end_day_key + today_key 都钉在 day_key 上：补算历史日时，判定窗口与基准日
    # 必须是被补算的那一天，而不是磁盘上最新的那一天。
    daily_results = load_daily_results(
        elder_id, n_days=required_history_days(config), end_day_key=day_key
    )
    result = judge_risk_level(elder_id, daily_results, config, today_key=day_key)

    # 把今天的 qualifies 写回今天的推理日志（保留 inference 已写入的全部字段）。
    #
    # 写回条件只看"这条日志是不是今天的"，**不再看 qualifies 是否非空**：
    # 分类结果全 False 也是有效结论（今天三条规则都不达标），它同样需要落盘，
    # 否则次日的持续性统计读不到今天、只能按"无该字段"处理。旧写法把
    # "分类失败"和"分类出全 False"混为一谈，两者都不写回。
    today_log = next(
        (r for r in reversed(daily_results)
         if (r.get("day_key") or r.get("date")) == day_key),
        None,
    )
    if today_log is not None:
        today_log["risk_type_qualifies"] = result.get("risk_type_qualifies") or {}
        today_log["risk_level"] = result.get("risk_level", 0)
        save_daily_result(elder_id, day_key, today_log)
    else:
        logger.warning(
            f"  └─ {day_key} 的推理日志不存在，qualifies 无处写回；"
            f"次日的持续性统计将把这天当作缺日"
        )

    return result

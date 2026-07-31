"""
风险类型判定规则（三类，双轨）

| 风险类型         | 方向性逻辑                                          | 持续性门槛        |
|------------------|-----------------------------------------------------|-------------------|
| 睡眠稳定性偏离   | SE↓ 且 WASO↑，再叠加 sol↑/bed_exit↑/deep↓ 中 ≥1     | 连续 3 天         |
| 社会连接减弱     | copresence↓ 且 out_of_home↓ 且 activity↓（三项全中）| 7 天滚动窗内 ≥5 天 |
| 作息节律紊乱     | RA↓ 且 IV↑，再叠加 sleep_onset_clock 漂移           | 连续 5 天         |

每类都要求**方向性超标 + 幅度门槛 + 持续性门槛**三者同时满足，不做单点触发。

三条设计决定的理由：

1. **为什么社会连接要"三项全中"**
   没人来（copresence↓）+ 也不出门（out_of_home↓）+ 家里也不怎么动（activity↓），
   三条同时成立才是真正的社交退缩，误报空间很小。
   代价是灵敏度：只"没人来但仍照常出门"的情况不触发本预警——这类会被
   周报层的 copresence 趋势捕捉到。这是为压低误报付出的代价，因为面向单个老人
   长期运行时，反复误报会直接导致家属关掉通知，届时灵敏度多高都没意义。

2. **为什么社会连接用滚动窗而不是"连续 5 天"**
   严格连续 5 天必然跨越周末，而周末社交本来就多，会把真实的退缩打断成两截、
   永远凑不满 5 天。滚动窗能容纳周末起伏，同时保持"不是单日波动"的克制。

3. **为什么作息节律紊乱是跨轨规则**
   RA/IV 在社交轨，sleep_onset_clock 在睡眠轨。这一类的文献支撑最强
   （昼夜节律振幅下降 + 碎片化上升是被动传感中与抑郁症状关联最稳定的指标），
   所以值得跨轨取证。睡眠轨缺失时降级为"仅 RA↓ 且 IV↑ 且持续 7 天"——
   门槛从 5 天提到 7 天，用更长的持续性换取失去交叉验证后的可信度。
"""

from dataclasses import dataclass, field

import numpy as np

from src.baseline.scaler_utils import TRACK_SLEEP, TRACK_SOCIAL
from src.utils.status import is_evaluable


@dataclass
class RiskRule:
    """
    单条风险判定规则。

    required: 必须全部方向性超标的特征
    optional: 至少满足 min_optional 项的特征（为空则不要求）
    """
    name: str
    key: str
    required: list[tuple[str, str, str]]          # (track, feature, direction)
    optional: list[tuple[str, str, str]] = field(default_factory=list)
    min_optional: int = 0
    threshold_ratio: float = 1.5                  # signed_z 的超标倍数门槛
    consecutive_days: int = 3                     # 连续达标天数门槛
    rolling_window: int | None = None             # 非 None 时用滚动窗替代连续判定
    rolling_required: int | None = None
    # 当某轨数据缺失导致 optional 池为空时，用更长的持续性门槛补偿
    consecutive_days_degraded: int | None = None

    def uses_rolling(self) -> bool:
        return self.rolling_window is not None and self.rolling_required is not None

    def all_features(self) -> list[tuple[str, str, str]]:
        return self.required + self.optional


def build_risk_rules(config: dict | None = None) -> dict[str, RiskRule]:
    """从配置构建三条规则。持续性门槛来自 config.risk.risk_rules。"""
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    rr = config.get("risk", {}).get("risk_rules", {})
    sleep_cfg = rr.get("sleep_stability", {})
    social_cfg = rr.get("social_decline", {})
    circ_cfg = rr.get("circadian_disruption", {})

    return {
        "sleep_stability": RiskRule(
            name="睡眠稳定性偏离",
            key="sleep_stability",
            required=[
                (TRACK_SLEEP, "sleep_efficiency", "down"),
                (TRACK_SLEEP, "waso_min", "up"),
            ],
            optional=[
                (TRACK_SLEEP, "sol_min", "up"),
                (TRACK_SLEEP, "bed_exit_count", "up"),
                (TRACK_SLEEP, "deep_sleep_ratio", "down"),
            ],
            min_optional=1,
            threshold_ratio=sleep_cfg.get("threshold_ratio", 1.5),
            consecutive_days=sleep_cfg.get("consecutive_days", 3),
        ),
        "social_decline": RiskRule(
            name="社会连接减弱",
            key="social_decline",
            required=[
                (TRACK_SOCIAL, "copresence_min", "down"),
                (TRACK_SOCIAL, "out_of_home_min", "down"),
                (TRACK_SOCIAL, "activity_counts", "down"),
            ],
            optional=[],
            min_optional=0,
            # 1.2 而非 1.5：三项全中且无可选池，门槛可低而联合误报率仍极低。
            # 另一层原因是 out_of_home_min 的残差 std 被周末双峰抬高（见 settings.yaml 注释）。
            threshold_ratio=social_cfg.get("threshold_ratio", 1.2),
            consecutive_days=social_cfg.get("rolling_required", 5),
            rolling_window=social_cfg.get("rolling_window", 7),
            rolling_required=social_cfg.get("rolling_required", 5),
        ),
        "circadian_disruption": RiskRule(
            name="作息节律紊乱",
            key="circadian_disruption",
            required=[
                (TRACK_SOCIAL, "rar_amplitude", "down"),
                (TRACK_SOCIAL, "rar_iv", "up"),
            ],
            optional=[
                (TRACK_SLEEP, "sleep_onset_clock", "any"),
            ],
            min_optional=1,
            threshold_ratio=circ_cfg.get("threshold_ratio", 1.5),
            consecutive_days=circ_cfg.get("consecutive_days", 5),
            consecutive_days_degraded=circ_cfg.get("consecutive_days_degraded", 7),
        ),
    }


def _exceeds(signed_z: float, direction: str, threshold: float) -> bool:
    """
    带符号方向判定：up 只认正向超标，down 只认负向超标，any 认双向。

    这是"方向匹配"的落点，保证"睡眠变好""活动变多"这类反向偏离不被计入风险。
    """
    if direction == "up":
        return signed_z > threshold
    if direction == "down":
        return signed_z < -threshold
    if direction == "any":
        return abs(signed_z) > threshold
    raise ValueError(f"Unknown direction: {direction!r}")


def _collect_signed_z(track_results: dict) -> tuple[dict, set[str]]:
    """
    从双轨推理结果里汇总 {(track, feature): signed_z}，并记录哪些轨可用。

    某轨 status 不是 success/observation，或 signed 统计不可用（旧格式基线）时，
    该轨整体视为不可用——不参与方向判定，而不是当成"没超标"。
    这个区分很重要：把"测不到"当成"正常"会掩盖设备故障。
    """
    z_map: dict[tuple[str, str], float] = {}
    available: set[str] = set()

    for track in (TRACK_SLEEP, TRACK_SOCIAL):
        tr = track_results.get(track)
        if not isinstance(tr, dict):
            continue
        if not is_evaluable(tr.get("status")):
            continue
        if not tr.get("signed_available", False):
            continue
        signed_z = tr.get("signed_z") or {}
        if not signed_z:
            continue
        available.add(track)
        for feat, value in signed_z.items():
            z_map[(track, feat)] = float(value)

    return z_map, available


def classify_risk_type(
    track_results: dict,
    daily_results: list[dict] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """
    根据当日双轨残差判断风险类型。

    Args:
        track_results: daily_inference 的返回（含 "sleep" / "social" 两键）
        daily_results: 近 N 天推理结果（用于统计连续/滚动天数）
        config: 全局配置

    Returns:
        [
            {
                "risk_type": "睡眠稳定性偏离",
                "risk_key": "sleep_stability",
                "score": 2.3,
                "is_active": True,
                "qualifies": True,
                "exceeding_features": ["sleep_efficiency", "waso_min", "sol_min"],
                "consecutive_days": 3,
                "threshold_required": 3,
                "evaluable": True,
                "skip_reason": None,
            },
            ...
        ]
    """
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    rules = build_risk_rules(config)
    z_map, available_tracks = _collect_signed_z(track_results)
    max_skip = _max_skip_days(config)
    history_days = required_history_days(config)

    results = []

    for key, rule in rules.items():
        needed_tracks = {t for t, _, _ in rule.required}
        missing_required_tracks = needed_tracks - available_tracks

        # 必需轨缺失 → 该类型本日不可评估（不是"正常"）
        if missing_required_tracks:
            results.append({
                "risk_type": rule.name,
                "risk_key": key,
                "score": 0.0,
                "is_active": False,
                "qualifies": False,
                "exceeding_features": [],
                "consecutive_days": 0,
                "threshold_required": rule.consecutive_days,
                "evaluable": False,
                "skip_reason": f"必需轨不可用: {sorted(missing_required_tracks)}",
            })
            continue

        threshold = rule.threshold_ratio

        # 必需项：全部都要方向性超标
        required_hits, required_z = [], []
        for track, feat, direction in rule.required:
            z = z_map.get((track, feat))
            if z is None:
                continue
            required_z.append(abs(z))
            if _exceeds(z, direction, threshold):
                required_hits.append(feat)

        all_required_met = len(required_hits) == len(rule.required)

        # 可选项：只统计所在轨可用的
        optional_hits, optional_z = [], []
        optional_evaluable = 0
        for track, feat, direction in rule.optional:
            if track not in available_tracks:
                continue
            z = z_map.get((track, feat))
            if z is None:
                continue
            optional_evaluable += 1
            optional_z.append(abs(z))
            if _exceeds(z, direction, threshold):
                optional_hits.append(feat)

        # 可选池为空（如睡眠轨离线导致 sleep_onset_clock 拿不到）→ 用降级门槛
        degraded = rule.min_optional > 0 and optional_evaluable == 0
        if degraded and rule.consecutive_days_degraded is not None:
            required_days = rule.consecutive_days_degraded
            optional_met = True   # 免除可选项要求，改用更长的持续性补偿
        elif degraded:
            required_days = rule.consecutive_days
            optional_met = False  # 无降级路径的规则：可选池空则无法达标
        else:
            required_days = rule.consecutive_days
            optional_met = len(optional_hits) >= rule.min_optional

        # 幅度分：全部参与特征的 |z| 加权平均（此处等权，权重已体现在 anomaly_score）
        all_z = required_z + optional_z
        score = float(np.mean(all_z)) if all_z else 0.0

        qualifies_today = bool(all_required_met and optional_met and score > 1.0)

        # 持续性统计。质量门只看**必需轨**：该规则的结论由必需轨决定，
        # 可选轨缺失走的是降级门槛（上面的 degraded 分支），不是质量问题。
        quality_tracks = frozenset(needed_tracks)
        if rule.uses_rolling():
            cons_days = _count_rolling_qualifies(
                key, daily_results, qualifies_today, rule.rolling_window,
                tracks=quality_tracks, max_skip=max_skip,
            )
            required_days = rule.rolling_required
        else:
            cons_days = _count_consecutive_qualifies(
                key, daily_results, qualifies_today, tracks=quality_tracks,
                max_steps=history_days, max_skip=max_skip,
            )

        is_active = bool(qualifies_today and cons_days >= required_days)

        results.append({
            "risk_type": rule.name,
            "risk_key": key,
            "score": round(score, 4),
            "is_active": is_active,
            "qualifies": qualifies_today,
            "exceeding_features": required_hits + optional_hits,
            "consecutive_days": cons_days,
            "threshold_required": required_days,
            "evaluable": True,
            "degraded_mode": degraded,
            "skip_reason": None,
        })

    return results


def _day_qualifies(day_result: dict, risk_key: str) -> bool:
    """某历史日志里，该风险类型当天是否达标（方向+幅度，不含持续性）。

    权威来源是 judge 写回的 `risk_type_qualifies` 字典。
    """
    quals = day_result.get("risk_type_qualifies")
    if isinstance(quals, dict) and risk_key in quals:
        return bool(quals[risk_key])
    for rt in day_result.get("risk_types", []) or []:
        if isinstance(rt, dict) and rt.get("risk_key") == risk_key:
            return bool(rt.get("qualifies", rt.get("is_active", False)))
    return False


def _counts_toward_consecutive(day_result: dict, tracks: frozenset[str] = frozenset()) -> bool:
    """
    该历史日是否计入持续性统计。

    degraded / insufficient / offline 的日子被**跳过**（既不累加也不打断）——
    传感器抖一下不该让攒了 4 天的偏离段清零，也不该凭空算成偏离。

    ★ 按轨判定，不看单一顶层值。质量本来就是按轨的：睡眠轨降级不该压住社交轨的
    持续性计数，那会把双轨的故障隔离在判定层又粘回去。只看**该规则必需的那些轨**，
    全部 valid 才计入。

    三级回退（顺序即优先级）：
        1. day_result[track]["data_quality"]  ← 现行格式
        2. day_result["data_quality"]         ← 单测桩与早期日志的单一顶层值
        3. 都没有 → True                       ← 历史日志无该字段，保守当作有效
    """
    per_track = [
        day_result[t].get("data_quality")
        for t in tracks
        if isinstance(day_result.get(t), dict)
        and day_result[t].get("data_quality") is not None
    ]
    if per_track:
        return all(q == "valid" for q in per_track)

    quality = day_result.get("data_quality")
    if quality is None:
        return True
    return quality == "valid"


def _history_before_today(daily_results: list[dict] | None) -> list[dict]:
    """
    取"今天之前"的历史日志。

    今天的达标信号必须用现算值——今天的日志此刻还没写回 risk_type_qualifies，
    读日志会漏掉今天，导致永远数不到自己。
    """
    if not daily_results:
        return []
    return daily_results[:-1]


def _day_key_of(day_result: dict) -> str | None:
    return day_result.get("day_key") or day_result.get("date")


def _index_by_day(daily_results: list[dict] | None) -> tuple[dict[str, dict], str | None]:
    """
    把历史日志按 day_key 建索引，并返回"今天"的 day_key。

    今天 = daily_results 的最后一条（调用方保证按日期升序）。拿不到日期的日志
    （极早期格式）不进索引，由回溯逻辑当作"缺日"处理。
    """
    if not daily_results:
        return {}, None
    today_key = _day_key_of(daily_results[-1])
    index = {}
    for day_result in _history_before_today(daily_results):
        key = _day_key_of(day_result)
        if key:
            index[key] = day_result
    return index, today_key


def _walk_back(
    index: dict[str, dict],
    today_key: str,
    max_steps: int,
    max_skip: int,
    tracks: frozenset[str],
):
    """
    从今天往前**逐个自然日**回溯，yield 计入统计的历史日（跳过不计入的）。

    ★ 为什么必须按自然日而不是按记录序号

      load_daily_results 取的是最近 N 个**文件**，不是最近 N 个**自然日**；
      而两轨全不可用那天 daily_job 直接跳过推理、连日志都不生成。两者叠加的
      结果是：设备离线一段时间后，断裂两端的偏离日会被当成连续日拼起来。
      实测 5 条日志跨越 22 个日历日（中间断 17 天）仍数出 consecutive=5 → L3。

    ★ 缺日与降级日走同一条路径

      两者都是"这天我们不知道"：缺日是压根没测，降级是测得不可信。既不累加
      也不打断，与 validator 的四态设计一致。但不能无上限地跨过去——连续跳过
      超过 max_skip 天就打断，否则一次长时间离线又会把两段无关的偏离粘起来。
      max_skip 默认 3，与 imputer.impute_sequence 的 max_forward_days 同源：
      超过三天没有可信数据，就不该再假装这是同一段状态。
    """
    from datetime import datetime, timedelta

    try:
        cursor = datetime.strptime(today_key, "%Y-%m-%d")
    except (TypeError, ValueError):
        return

    skipped_in_a_row = 0
    for _ in range(max_steps):
        cursor -= timedelta(days=1)
        key = cursor.strftime("%Y-%m-%d")
        day_result = index.get(key)

        if day_result is None or not _counts_toward_consecutive(day_result, tracks):
            skipped_in_a_row += 1
            if skipped_in_a_row > max_skip:
                return          # 连续跳过太久，不再认为是同一段状态
            continue

        skipped_in_a_row = 0
        yield day_result


def _count_consecutive_qualifies(
    risk_key: str,
    daily_results: list[dict] | None,
    today_qualifies: bool,
    tracks: frozenset[str] = frozenset(),
    max_steps: int = 30,
    max_skip: int = 3,
) -> int:
    """
    统计截至今天的连续达标天数（按自然日回溯）。

    规则：今天不达标直接返回 0；往前逐日回溯，跳过质量不佳的日子与缺日，
    遇到第一个"质量正常但不达标"的日子即停。
    """
    if not today_qualifies:
        return 0

    index, today_key = _index_by_day(daily_results)
    if not today_key:
        return 1

    count = 1
    for day_result in _walk_back(index, today_key, max_steps, max_skip, tracks):
        if _day_qualifies(day_result, risk_key):
            count += 1
        else:
            break
    return count


def _count_rolling_qualifies(
    risk_key: str,
    daily_results: list[dict] | None,
    today_qualifies: bool,
    window: int,
    tracks: frozenset[str] = frozenset(),
    max_skip: int = 3,
) -> int:
    """
    统计 window 个**自然日**滚动窗内的达标天数（含今天）。

    窗口按日历定长，不随记录条数伸缩：旧实现取"最近 window-1 条历史记录"，
    日志有空洞时窗口会悄悄拉长到几周。
    """
    count = 1 if today_qualifies else 0

    index, today_key = _index_by_day(daily_results)
    if not today_key or window <= 1:
        return count

    # 今天占 1 天，往前再看 window-1 个自然日
    for day_result in _walk_back(index, today_key, window - 1, max_skip, tracks):
        if _day_qualifies(day_result, risk_key):
            count += 1
    return count


def required_history_days(config: dict | None = None) -> int:
    """
    判定层需要加载多少天推理日志。

    不能写死 7：circadian 降级模式要求连续 7 天达标，而"今天 + 6 天历史"恰好
    只有 7 条——历史里只要有一个 degraded 日被跳过，计数上限就掉到 6，规则在
    数学上永远无法激活。而降级模式**正是因为睡眠轨不可用才进入的**，恰恰是
    最容易伴随数据质量问题的场景。

    最坏情况 = 最长门槛 + 每两个达标日之间都插满 max_skip 天跳过。
    留 30 天上限：再长也没有统计意义，且日志读取是每天一个小 JSON，成本可忽略。
    """
    rules = build_risk_rules(config)
    max_skip = _max_skip_days(config)

    longest = 1
    for rule in rules.values():
        need = rule.rolling_window if rule.uses_rolling() else max(
            rule.consecutive_days, rule.consecutive_days_degraded or 0
        )
        longest = max(longest, need)

    return min(longest + (longest - 1) * max_skip, 30)


def _max_skip_days(config: dict | None = None) -> int:
    """连续跳过多少天后打断持续性统计。"""
    if config is None:
        from src.utils.io import load_config
        config = load_config()
    return config.get("risk", {}).get("continuity", {}).get("max_skip_days", 3)


def get_risk_feature_importance(risk_key: str, config: dict | None = None) -> dict[str, float]:
    """获取某个风险类型各特征的相对重要性（按 feature_weights.json 的权重归一）"""
    from src.utils.io import get_feature_weight_map

    rules = build_risk_rules(config)
    rule = rules.get(risk_key)
    if rule is None:
        return {}

    weight_cache: dict[str, dict[str, float]] = {}
    entries: dict[str, float] = {}
    for track, feat, _ in rule.all_features():
        if track not in weight_cache:
            weight_cache[track] = get_feature_weight_map(track)
        entries[feat] = weight_cache[track].get(feat, 1.0)

    total = sum(entries.values())
    if total <= 0:
        return {}
    return {feat: w / total for feat, w in entries.items()}


def list_risk_types(config: dict | None = None) -> list[str]:
    """列出所有风险类型名称"""
    return [rule.name for rule in build_risk_rules(config).values()]

"""
小贝壳无感睡眠监测仪适配器（睡眠轨唯一主源）

职责：睡眠分期时间轴 → 8 维睡眠特征。

★ 本模块承载 day_key 时间窗派生逻辑，是"night_of 循环定义"的修法所在。

    错误做法（原设计文档 5.1）：
        night_of      = 入睡时刻所在自然日
        搜索窗口       = [night_of 20:00, night_of+1 12:00]
    这是循环定义——要先知道 night_of 才能定搜索窗口，要先有搜索窗口才能找入睡时刻。
    更糟的是：入睡时刻从 23:30 漂到 00:30 时，night_of 从 D 跳到 D+1，而 00:30 并不
    落在 [D+1 20:00, D+2 12:00] 里，定义直接失效。而就寝相位漂移正是
    sleep_onset_clock 要监测的量——索引键绝不能绑在被监测量上，否则会在最该观察的
    那几天制造一个重复日 + 一个空洞日。

    本模块的做法：
        day_key   = 自然日 D（纯日历，与任何测量值无关）
        搜索区间   = [D−1 18:00, D 14:00]   固定，不依赖入睡时刻
        睡眠特征(D) ← 在 D 日早晨结束的那一夜
    入睡时刻仍可自由漂移（含跨午夜），只影响 sleep_onset_clock 的值，不影响索引。

双后端归一化：设备可能绑定 huayi 或 whst（森思泰克）后端，字段完全不同。
两条路径都映射到同一套内部字段名，上层不感知差异。
部署期必须用真实设备序列号探测走哪条路径——**不得用接口失败来推断后端**
（失败可能只是网络问题，据此切后端会静默用错口径）。
"""

from datetime import datetime, time, timedelta

import numpy as np

from src.baseline.scaler_utils import SLEEP_FEATURES
from src.data_pipeline.adapters import SensorAdapter

# 睡眠分期枚举（huayi 口径）
STAGE_DEEP = "DEEP"
STAGE_LIGHT = "LIGHT"
STAGE_MIDDLE = "MIDDLE"
STAGE_WAKEFUL = "WAKEFUL"
STAGE_UNKNOWN = "UNKNOWN"      # ★ 语义是"离床"，不是"未知分期"

# 计入 TST（总睡眠时长）的分期
SLEEP_STAGES = frozenset({STAGE_LIGHT, STAGE_MIDDLE, STAGE_DEEP})
# 在床状态：除 UNKNOWN 外都算在床
IN_BED_STAGES = frozenset({STAGE_LIGHT, STAGE_MIDDLE, STAGE_DEEP, STAGE_WAKEFUL})

# 搜索区间：前一日 18:00 → 当日 14:00
SEARCH_START_HOUR = 18
SEARCH_END_HOUR = 14

# 起床判据：最后一个在床段之后离床持续超过此值才算真起床（否则是夜间如厕）
TRUE_RISE_GAP_MIN = 30

# sleep_onset_clock 的参考零点：20:00。晚于 20:00 为正，早于为负。
ONSET_REFERENCE_HOUR = 20


def search_window(day_key: str) -> tuple[datetime, datetime]:
    """
    某 day_key 对应的固定搜索区间 [D−1 18:00, D 14:00]。

    纯日历计算，不依赖任何测量值——这是修掉循环定义的关键。
    """
    d = datetime.strptime(day_key, "%Y-%m-%d")
    start = datetime.combine(d.date() - timedelta(days=1), time(SEARCH_START_HOUR, 0))
    end = datetime.combine(d.date(), time(SEARCH_END_HOUR, 0))
    return start, end


def onset_clock_minutes(onset: datetime) -> float:
    """
    入睡时刻编码为距参考日 20:00 的分钟偏移。

    参考日取"入睡时刻所在那个夜晚的傍晚"：若入睡在午夜后（0-14 点），
    参考点是前一天 20:00，于是跨午夜入睡得到 >240 的正值而不是负值突变。

    例：22:30 入睡 → +150；00:30 入睡 → +270；19:30 入睡 → −30。
    值连续单调，正是 sleep_onset_clock 要捕捉的相位漂移，不做截断。
    """
    ref_date = onset.date()
    if onset.hour < SEARCH_END_HOUR:
        ref_date = ref_date - timedelta(days=1)
    ref = datetime.combine(ref_date, time(ONSET_REFERENCE_HOUR, 0))
    return (onset - ref).total_seconds() / 60.0


def _parse_timeline(timeline: list[dict]) -> list[tuple[datetime, str]]:
    """把 [{stage, ts}] 解析成按时间升序的 (时刻, 分期) 列表，脏记录跳过"""
    parsed: list[tuple[datetime, str]] = []
    for item in timeline or []:
        ts, stage = item.get("ts"), item.get("stage")
        if ts is None or not stage:
            continue
        try:
            dt = datetime.fromisoformat(ts) if isinstance(ts, str) else \
                datetime.fromtimestamp(float(ts) / 1000 if float(ts) > 1e11 else float(ts))
        except (ValueError, TypeError, OSError):
            continue
        parsed.append((dt, str(stage).upper()))
    parsed.sort(key=lambda x: x[0])
    return parsed


def _segments(
    parsed: list[tuple[datetime, str]],
    window_end: datetime,
) -> list[tuple[datetime, datetime, str]]:
    """
    把时刻点序列转成 (起, 止, 分期) 段。最后一段延伸到下一个点或窗口末尾。

    小贝壳返回的是"分期变化点"，段时长要靠相邻点相减得出。
    """
    segs = []
    for i, (start, stage) in enumerate(parsed):
        end = parsed[i + 1][0] if i + 1 < len(parsed) else window_end
        if end > start:
            segs.append((start, end, stage))
    return segs


def derive_sleep_features(
    day_key: str,
    timeline: list[dict],
    heart_minutes: list[dict] | None = None,
) -> dict:
    """
    从睡眠分期时间轴派生 8 维睡眠特征。

    Args:
        day_key: 自然日 "YYYY-MM-DD"，特征归属日
        timeline: [{"stage": "LIGHT", "ts": "2026-08-01T23:10:00"}, ...]
        heart_minutes: [{"avg": 62.0, "ts": "..."}] 每 10 分钟心率，用于 night_hr_mean

    Returns:
        {feature_name: value | None}，无法计算的特征为 None

    派生规则：
        T_bed   = 搜索区间内首个在床段的起点
        T_onset = T_bed 之后首个睡眠期的时刻
        T_rise  = 最后一个在床段的终点（其后离床 >30 min 才算真起床）
        TIB     = T_rise − T_bed
        TST     = Σ 睡眠期时长 in [T_bed, T_rise]
        WASO    = Σ WAKEFUL 时长 in [T_onset, T_rise]
        SOL     = T_onset − T_bed
        离床次数 = [T_onset, T_rise] 内的 UNKNOWN 段数
        日间小睡 = T_rise 之后（仍在搜索区间内）的睡眠期时长
    """
    features: dict[str, float | None] = {name: None for name in SLEEP_FEATURES}

    win_start, win_end = search_window(day_key)
    parsed = [(dt, st) for dt, st in _parse_timeline(timeline) if win_start <= dt <= win_end]
    if not parsed:
        return features

    segs = _segments(parsed, win_end)
    in_bed = [s for s in segs if s[2] in IN_BED_STAGES]
    if not in_bed:
        return features

    t_bed = in_bed[0][0]

    # 真起床：从 t_bed 往后走，遇到第一个"离床 >30 min"的间隔就停——
    # 那之前的最后一个在床段终点即 T_rise。
    #
    # 为什么必须在第一个长间隔处停：日间小睡也是"在床段"，若一路走到最后一段，
    # T_rise 会被推到下午小睡的结束时刻，于是整个小睡被算进夜间 TIB
    # （睡眠效率被稀释），而 daytime_nap_min 反而恒为 0。
    # 短于 30 min 的离床是夜间如厕/游走，计入 bed_exit_count 而不算起床。
    t_rise = in_bed[0][1]
    for i in range(1, len(in_bed)):
        gap = (in_bed[i][0] - in_bed[i - 1][1]).total_seconds() / 60.0
        if gap > TRUE_RISE_GAP_MIN:
            break
        t_rise = in_bed[i][1]

    # 入睡时刻：T_bed 之后首个睡眠期
    t_onset = None
    for start, _, stage in segs:
        if start >= t_bed and stage in SLEEP_STAGES:
            t_onset = start
            break

    def _minutes_in(stages: frozenset, lo: datetime, hi: datetime) -> float:
        total = 0.0
        for s_start, s_end, stage in segs:
            if stage not in stages:
                continue
            overlap_start, overlap_end = max(s_start, lo), min(s_end, hi)
            if overlap_end > overlap_start:
                total += (overlap_end - overlap_start).total_seconds() / 60.0
        return total

    tib_min = (t_rise - t_bed).total_seconds() / 60.0
    tst_min = _minutes_in(SLEEP_STAGES, t_bed, t_rise)
    deep_min = _minutes_in(frozenset({STAGE_DEEP}), t_bed, t_rise)

    if tib_min > 0:
        features["sleep_efficiency"] = float(np.clip(tst_min / tib_min, 0.0, 1.0))
    if tst_min > 0:
        features["deep_sleep_ratio"] = float(np.clip(deep_min / tst_min, 0.0, 1.0))

    if t_onset is not None:
        features["sol_min"] = max(0.0, (t_onset - t_bed).total_seconds() / 60.0)
        features["sleep_onset_clock"] = onset_clock_minutes(t_onset)
        features["waso_min"] = _minutes_in(frozenset({STAGE_WAKEFUL}), t_onset, t_rise)
        # 离床次数：入睡后到起床前的 UNKNOWN 段数
        features["bed_exit_count"] = float(sum(
            1 for s_start, s_end, stage in segs
            if stage == STAGE_UNKNOWN and t_onset <= s_start < t_rise
        ))

    # 日间小睡：真起床之后仍在搜索区间内的睡眠期
    features["daytime_nap_min"] = _minutes_in(SLEEP_STAGES, t_rise, win_end)

    # 夜间平均心率
    # ⚠️ 这是 hrv_rmssd 的**弱代理**，不等价于 HRV。小贝壳只给 BPM 级聚合值，
    # 没有逐拍 R-R 间期，RMSSD 在数学上无法计算。对外措辞不得写成 HRV。
    if heart_minutes and t_onset is not None:
        values = []
        for item in heart_minutes:
            ts, avg = item.get("ts"), item.get("avg")
            if ts is None or avg is None:
                continue
            try:
                dt = datetime.fromisoformat(ts) if isinstance(ts, str) else \
                    datetime.fromtimestamp(float(ts))
            except (ValueError, TypeError, OSError):
                continue
            if t_onset <= dt <= t_rise:
                values.append(float(avg))
        if values:
            features["night_hr_mean"] = float(np.mean(values))

    return features


def normalize_whst_report(report: dict) -> list[dict]:
    """
    森思泰克（whst）后端的睡眠报告 → huayi 口径的分期时间轴。

    whst 直接给上床/入睡/醒来/起床时刻与各阶段时长，没有逐段时间轴。
    这里合成一条等价时间轴，使下游 derive_sleep_features 无需分支。

    ⚠️ 合成时间轴的分期顺序是构造的（浅→深→浅→清醒），不是真实测量顺序。
    因此 whst 后端下 deep_sleep_ratio 的**分期时序信息不可用**，
    但比例与总时长是真实的——这正是特征只用比例、不用时序的原因。
    """
    def _parse(key: str) -> datetime | None:
        value = report.get(key)
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    t_bed = _parse("inBedTime") or _parse("bedTime")
    t_onset = _parse("sleepTime") or _parse("fallAsleepTime")
    t_wake = _parse("wakeTime")
    t_rise = _parse("outBedTime") or _parse("getUpTime")

    if not (t_bed and t_rise):
        return []

    timeline = [{"stage": STAGE_WAKEFUL, "ts": t_bed.isoformat()}]
    if t_onset:
        light = float(report.get("lightSleepDuration", 0) or 0)
        deep = float(report.get("deepSleepDuration", 0) or 0)
        cursor = t_onset
        # 构造顺序：浅睡一半 → 深睡全部 → 浅睡另一半
        for stage, minutes in (
            (STAGE_LIGHT, light / 2), (STAGE_DEEP, deep), (STAGE_LIGHT, light / 2)
        ):
            if minutes > 0:
                timeline.append({"stage": stage, "ts": cursor.isoformat()})
                cursor += timedelta(minutes=minutes)
        if t_wake and t_wake > cursor:
            timeline.append({"stage": STAGE_WAKEFUL, "ts": t_wake.isoformat()})

    timeline.append({"stage": STAGE_UNKNOWN, "ts": t_rise.isoformat()})
    return timeline


class XiaobeikeAdapter(SensorAdapter):
    """
    小贝壳无感睡眠监测仪 → 8 维睡眠特征。

    Usage:
        adapter = XiaobeikeAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")

        # 接入真实设备（需先完成 §11 的待决问题）
        adapter = XiaobeikeAdapter(mode="live", backend="huayi")
        features = adapter.extract(source="<deviceSerial>", date="2026-08-01")
    """

    FEATURE_NAMES = list(SLEEP_FEATURES)
    BACKENDS = ("huayi", "whst")

    def __init__(self, mode: str = "mock", backend: str = "huayi"):
        super().__init__(mode)
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got {backend!r}")
        self.backend = backend

    def _read_raw(self, source: str, date: str) -> dict:
        """
        【接入真实设备时实现】拉取当日睡眠数据并派生特征。

        huayi 路径：
            GET /analysis/v1/devices/{deviceId}/daily/sleep?date=      → 分期时间轴
            GET /analysis/v1/devices/{deviceId}/daily/average/hearts   → 每10min心率
        whst 路径：
            GET /sleepReport/list?startDate=&endDate=&deviceSerial=    → 报告（经 normalize_whst_report 转换）
            GET /statistics/data/bodyDetect?deviceSerial=              → 在离床（messageType 1=在床 2=离床）

        实现要点：
          1. 先用 deviceSerial 换 deviceId，后续接口都用 deviceId
          2. 拉取区间用 search_window(date) 的结果，不要用自然日 00:00-24:00
          3. huayi 与 whst 的日界方向可能相反，必须用真实设备实测确认（§11 待决问题）
          4. 拿到 timeline 后调 derive_sleep_features(date, timeline, heart_minutes)
        """
        raise NotImplementedError(
            f"小贝壳 live 模式（backend={self.backend}）尚未接入。"
            f"依赖 §11 待决问题：设备形态（床垫下/床边）与日界方向需真机实测。"
            f"已就绪的部分：derive_sleep_features() 的时间窗派生逻辑与 whst 归一化。"
        )

    def _generate_mock(self, date: str) -> dict:
        """生成一条合成时间轴并走真实派生逻辑（而非直接编造特征值）。

        走同一条派生路径的好处：mock 数据也会经过 derive_sleep_features 的
        时间窗计算，能顺带验证派生逻辑没坏。
        """
        import random

        rng = random.Random(date)
        win_start, _ = search_window(date)
        bed_dt = win_start + timedelta(minutes=rng.randint(240, 330))   # 22:00-23:30
        onset_dt = bed_dt + timedelta(minutes=rng.randint(10, 30))

        timeline = [{"stage": STAGE_WAKEFUL, "ts": bed_dt.isoformat()}]
        cursor = onset_dt
        for stage, dur in (
            (STAGE_LIGHT, 90), (STAGE_DEEP, 70), (STAGE_WAKEFUL, rng.randint(10, 30)),
            (STAGE_UNKNOWN, rng.randint(5, 12)), (STAGE_LIGHT, 110), (STAGE_DEEP, 45),
        ):
            timeline.append({"stage": stage, "ts": cursor.isoformat()})
            cursor += timedelta(minutes=dur)
        rise_dt = cursor
        timeline.append({"stage": STAGE_UNKNOWN, "ts": rise_dt.isoformat()})

        hearts = [
            {"avg": 60 + rng.gauss(0, 3), "ts": (onset_dt + timedelta(minutes=10 * i)).isoformat()}
            for i in range(24)
        ]

        features = derive_sleep_features(date, timeline, hearts)
        return {k: v for k, v in features.items() if v is not None}

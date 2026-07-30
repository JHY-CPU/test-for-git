"""
萤石告警事件适配器（社会连接轨的活动与节律来源）

职责：C6c 移动/人形事件 + T1C PIR 事件 → 小时活动序列 a[h] → 4 维特征
      （activity_counts / out_of_home_min / rar_amplitude / rar_iv）
      copresence_min 由 camera.py 单独产出（边缘人形共处检测）。

★ 7 天保留期是硬约束
萤石告警消息列表只保留最近 7 天，且 T1C 没有"取一段时间内触发次数"的统计接口——
告警列表是唯一的历史事件来源。拉取任务连续失败 7 天即造成**不可恢复**的数据空洞。
因此：每日必须落本地库，且拉取失败 2 天就要告警运维（不是 7 天）。

告警类型分流（入 a[h] vs 仅落库）见 ALARM_TYPES_ACTIVITY / ALARM_TYPES_SAFETY_ONLY：
跌倒与未活动报警走安全告警通道，与心理监测无关，绝不能混进活动量——
否则一次跌倒会被读成"活动量上升"。
"""

from datetime import datetime, time, timedelta

import numpy as np

from src.data_pipeline.adapters import SensorAdapter
from src.data_pipeline.adapters.circadian import (
    HOURS_PER_DAY,
    build_hourly_counts,
    compute_circadian_features,
)

# 入 a[h] 活动序列的告警类型
ALARM_TYPES_ACTIVITY = {
    10000: "人体感应事件(T1C)",
    10002: "移动侦测(C6c)",
    10120: "智能人形检测(C6c)",
    10140: "有人移动(T1C)",
    10147: "有人出现(C6c)",
    10148: "有人停留(C6c)",
    10027: "有人徘徊(C6c)",
}

# 仅落库、不入模：安全告警通道
ALARM_TYPES_SAFETY_ONLY = {
    12259: "有人跌倒",
    12257: "未活动报警",
}

# v2.1 不使用：音频异常（GRU 管线完全不采集音频）
ALARM_TYPES_UNUSED = {10022: "音频异常"}

# 事件去重窗口：窗口内多次触发合并为一次
DEDUP_WINDOW_SEC = 60

# 疑似离家：PIR 与摄像头同时静默且非在床，持续 ≥此值
OUT_OF_HOME_MIN_GAP = 30

# T1C 电量低于此值触发降级（静默漏报会被误读成"活动量下降"）
LOW_BATTERY_THRESHOLD = 20


def _to_datetime(value) -> datetime | None:
    """萤石 alarmTime 是毫秒时间戳；也接受 ISO 字符串"""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        v = float(value)
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v)
    except (ValueError, TypeError, OSError):
        return None


def filter_activity_events(alarms: list[dict]) -> list[str]:
    """
    从告警列表里挑出应计入活动量的事件，返回 ISO 时间戳列表。

    只保留 ALARM_TYPES_ACTIVITY 中的类型。未知类型一律丢弃而非默认计入——
    萤石可能新增类型，默认计入会让活动量口径悄悄改变。
    """
    out = []
    for alarm in alarms or []:
        try:
            alarm_type = int(alarm.get("alarmType"))
        except (TypeError, ValueError):
            continue
        if alarm_type not in ALARM_TYPES_ACTIVITY:
            continue
        dt = _to_datetime(alarm.get("alarmTime"))
        if dt is not None:
            out.append(dt.isoformat())
    return out


def derive_day_window(
    day_key: str,
    rise_time: str | None = None,
    bed_time: str | None = None,
) -> tuple[datetime, datetime]:
    """
    日间窗口 = [T_rise, 当晚 T_bed]。

    拿不到睡眠轨的起床/上床时刻时，退回固定的 07:00-23:00。
    退化会让窗口边界与真实作息错开，但不会破坏 day_key 索引——
    这正是把索引与测量值解耦的好处：睡眠轨掉线只影响窗口精度，不影响归属日。
    """
    d = datetime.strptime(day_key, "%Y-%m-%d").date()
    rise = _to_datetime(rise_time) or datetime.combine(d, time(7, 0))
    bed = _to_datetime(bed_time) or datetime.combine(d, time(23, 0))
    if bed <= rise:
        bed = datetime.combine(d, time(23, 59))
    return rise, bed


def compute_out_of_home(
    event_times: list[str],
    window: tuple[datetime, datetime],
    in_bed_intervals: list[tuple[datetime, datetime]] | None = None,
    min_gap_min: int = OUT_OF_HOME_MIN_GAP,
) -> tuple[float, int]:
    """
    疑似离家时长：日间窗口内 PIR 与摄像头同时无事件、且非在床、持续 ≥min_gap 的时段总长。

    ⚠️ 命名必须保留"疑似"。已知误判源：老人在传感器盲区（阳台、厨房深处、卫生间）
    长时间静坐，或在沙发上午睡不动。单摄像头 + 单 PIR 无法排除。

    Returns:
        (out_of_home_min, gap_count) —— gap_count 供周报解释"分几次外出"
    """
    start, end = window
    times = sorted(t for t in (_to_datetime(x) for x in event_times) if t and start <= t <= end)

    def _in_bed_overlap(lo: datetime, hi: datetime) -> float:
        """静默段与在床时段的重叠分钟数"""
        overlap = 0.0
        for b_start, b_end in (in_bed_intervals or []):
            o_lo, o_hi = max(lo, b_start), min(hi, b_end)
            if o_hi > o_lo:
                overlap += (o_hi - o_lo).total_seconds() / 60.0
        return overlap

    boundaries = [start] + times + [end]
    total, count = 0.0, 0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        gap_min = (hi - lo).total_seconds() / 60.0

        # 扣掉在床部分，只留真正"既无事件又不在床"的时长。
        # 不用"整段是否被在床时段包含"来判断：静默段天然会比在床时段稍宽
        # （躺下前、起身后各有几分钟无事件），整段包含判据几乎永不成立，
        # 于是午睡仍会被整段算成离家。按重叠扣减才是正确语义，
        # 也顺带处理"上半段在床、下半段真外出"的混合情形。
        effective = gap_min - _in_bed_overlap(lo, hi)
        if effective >= min_gap_min:
            total += effective
            count += 1
    return total, count


def derive_social_activity_features(
    day_key: str,
    alarms: list[dict],
    rise_time: str | None = None,
    bed_time: str | None = None,
    in_bed_intervals: list[tuple[datetime, datetime]] | None = None,
    t1c_battery: int | None = None,
) -> dict:
    """
    告警事件 → 社会连接轨的 4 维活动/节律特征。

    Args:
        day_key: 自然日
        alarms: 萤石告警列表 [{"alarmType": 10002, "alarmTime": 1754006400000}, ...]
        rise_time / bed_time: 睡眠轨给出的起床/上床时刻（缺失时退回 07:00-23:00）
        in_bed_intervals: 在床时段，用于排除午睡被误算成离家
        t1c_battery: T1C 剩余电量百分比

    Returns:
        {
            "activity_counts": float,
            "out_of_home_min": float,
            "rar_amplitude": float,
            "rar_iv": float,
            "_hourly": list[float],       # 诊断：小时序列
            "_out_of_home_count": int,    # 周报：分几次外出
            "_device_degraded": bool,     # 电量低 → 上层标 degraded
        }
    """
    event_times = filter_activity_events(alarms)
    window = derive_day_window(day_key, rise_time, bed_time)

    # 小时序列覆盖整个自然日（RA/IV 是昼夜指标，必须看全 24 小时，不能只看日间窗口）
    hourly = build_hourly_counts(event_times, dedup_window_sec=DEDUP_WINDOW_SEC)
    circ = compute_circadian_features(hourly)

    # activity_counts 只数日间窗口内的事件（这是"日间活动量"的定义）
    start, end = window
    day_events = [
        t for t in (_to_datetime(x) for x in event_times) if t and start <= t <= end
    ]
    # 与 a[h] 同口径去重
    day_events.sort()
    deduped, last = 0, None
    for t in day_events:
        if last is None or (t - last).total_seconds() >= DEDUP_WINDOW_SEC:
            deduped += 1
            last = t

    out_min, out_count = compute_out_of_home(event_times, window, in_bed_intervals)

    return {
        "activity_counts": float(deduped),
        "out_of_home_min": float(out_min),
        "rar_amplitude": circ["rar_amplitude"],
        "rar_iv": circ["rar_iv"],
        "_hourly": [float(v) for v in hourly],
        "_m10_start_hour": circ["m10_start_hour"],
        "_l5_start_hour": circ["l5_start_hour"],
        "_out_of_home_count": out_count,
        "_device_degraded": bool(
            t1c_battery is not None and t1c_battery < LOW_BATTERY_THRESHOLD
        ),
    }


class EzvizEventAdapter(SensorAdapter):
    """
    萤石 C6c + T1C 告警事件 → 社会连接轨的活动/节律特征。

    Usage:
        adapter = EzvizEventAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")
    """

    FEATURE_NAMES = ["activity_counts", "out_of_home_min", "rar_amplitude", "rar_iv"]

    def _read_raw(self, source: str, date: str) -> dict:
        """
        【接入真实设备时实现】

            POST /api/lapp/token/get                → accessToken
            POST /api/lapp/alarm/device/list        → 告警列表（⚠️ 仅最近 7 天）
            GET  /api/v3/otap/prop/{serial}/global/0/PowerInfo/RemainingPower → T1C 电量

        实现要点：
          1. **每日必须落本地 SQLite**：7 天保留期过后云端数据永久消失
          2. 断点续拉 + 失败重试；连续失败 2 天即告警运维（不要等到 7 天）
          3. 用 filter_activity_events 分流告警类型，跌倒/未活动不得进活动量
          4. 布防状态会影响事件产生，需小时级巡检（§11 待决问题）
        """
        raise NotImplementedError(
            "萤石告警拉取 live 模式尚未接入。依赖 §11 待决问题："
            "webhook 验签方式、布防状态巡检、本地 SQLite 落库策略。"
            "已就绪的部分：derive_social_activity_features() 的事件→特征派生逻辑。"
        )

    def _generate_mock(self, date: str) -> dict:
        """生成合成告警列表并走真实派生逻辑"""
        import random

        rng = random.Random(f"{date}-ezviz")
        d = datetime.strptime(date, "%Y-%m-%d")
        profile = [0, 0, 0, 0, 0, 1, 3, 8, 11, 12, 10, 9, 8, 6, 5, 7, 10, 12, 11, 8, 5, 2, 1, 0]

        alarms = []
        for hour, count in enumerate(profile):
            for _ in range(count):
                ts = d + timedelta(
                    hours=hour, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
                )
                alarms.append({
                    "alarmType": rng.choice(list(ALARM_TYPES_ACTIVITY)),
                    "alarmTime": int(ts.timestamp() * 1000),
                })
        # 混入一个安全告警，验证它不会进活动量
        alarms.append({
            "alarmType": 12259,
            "alarmTime": int((d + timedelta(hours=15)).timestamp() * 1000),
        })

        derived = derive_social_activity_features(date, alarms)
        return {k: v for k, v in derived.items() if not k.startswith("_")}

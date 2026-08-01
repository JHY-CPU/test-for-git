"""
昼夜节律指标：由小时活动序列 a[0..23] 派生 M10 / L5 / RA / IV

两个入模特征：

    rar_amplitude  相对振幅 RA = (M10 − L5) / (M10 + L5)
        M10 = 活动量最高的连续 10 小时的均值
        L5  = 活动量最低的连续 5 小时的均值
        RA ∈ [0, 1]，越接近 1 表示昼夜节律越分明（白天活跃、夜间安静）。
        异常方向：↓（振幅塌陷 = 昼夜界限模糊）

    rar_iv         日内变异性 IV = n·Σ(a_h − a_{h−1})² / [(n−1)·Σ(a_h − ā)²]
        相邻小时差分的方差 / 全序列方差。越大表示活动越碎片化。
        异常方向：↑（碎片化上升）

⚠️ 测量链的诚实交代：文献 21 的昼夜节律证据全部来自腕表体动计（连续采样佩戴者
自身的肢体运动），而本方案用单室 PIR 事件计数——测的是"在被监测区域内的存在与
移动"。老人下午待在未覆盖的卧室，客厅 PIR 近零，会被读成 L5 低谷，RA 被系统性
扭曲。这条差异写在设计文档局限里，需要腕表并行实验来弥合。

M10/L5 的连续窗口是**环形**的：一天的 23 点和次日 0 点在生理上是连续的，
非环形实现会漏掉跨午夜的低谷窗口（比如 22:00-03:00 这个最安静的 5 小时）。
"""

import numpy as np

HOURS_PER_DAY = 24
M10_WINDOW = 10
L5_WINDOW = 5


def _validate_hourly(hourly_counts) -> np.ndarray:
    """把输入规整成长度 24 的 float 数组，顺带挡住常见的形状/取值错误。"""
    arr = np.asarray(hourly_counts, dtype=np.float64).flatten()
    if arr.shape[0] != HOURS_PER_DAY:
        raise ValueError(
            f"hourly_counts must have {HOURS_PER_DAY} entries, got {arr.shape[0]}"
        )
    if np.any(arr < 0):
        raise ValueError("hourly_counts must be non-negative")
    return arr


def _circular_window_means(arr: np.ndarray, width: int) -> np.ndarray:
    """
    所有环形连续窗口的均值，返回长度 24 的数组（下标 = 窗口起点小时）。

    用 np.concatenate 拼一份再滑窗，比逐窗口取模索引清晰且不易写错边界。
    """
    doubled = np.concatenate([arr, arr[: width - 1]]) if width > 1 else arr.copy()
    # 累积和做定宽滑窗求和：O(24)，比循环切片快，也避免重复求和的浮点漂移
    cumsum = np.concatenate([[0.0], np.cumsum(doubled)])
    sums = cumsum[width:width + HOURS_PER_DAY] - cumsum[:HOURS_PER_DAY]
    return sums / width


def compute_m10_l5(hourly_counts) -> tuple[float, float, int, int]:
    """
    计算 M10（最活跃连续 10 小时均值）与 L5（最安静连续 5 小时均值）。

    Returns:
        (m10, l5, m10_start_hour, l5_start_hour)
        起点小时一并返回，供周报解释"最活跃时段在几点"。
    """
    arr = _validate_hourly(hourly_counts)

    m10_means = _circular_window_means(arr, M10_WINDOW)
    l5_means = _circular_window_means(arr, L5_WINDOW)

    m10_start = int(np.argmax(m10_means))
    l5_start = int(np.argmin(l5_means))

    return (
        float(m10_means[m10_start]),
        float(l5_means[l5_start]),
        m10_start,
        l5_start,
    )


def compute_rar_amplitude(hourly_counts) -> float:
    """
    相对振幅 RA = (M10 − L5) / (M10 + L5)。

    整天零活动时分母为 0 —— 返回 NaN 而不是 0。这个区分很重要：
    RA=0 的语义是"昼夜完全没有差别"（老人在家但作息塌陷），
    NaN 的语义是"这天没采到任何活动事件"（设备离线/撤防）。
    两者对应完全不同的处置，混成 0 会让"设备坏了"被模型读成"作息塌陷"。
    """
    m10, l5, _, _ = compute_m10_l5(hourly_counts)
    total = m10 + l5
    if total <= 0:
        return float("nan")
    return float((m10 - l5) / total)


def compute_rar_iv(hourly_counts) -> float:
    """
    日内变异性 IV = n·Σ(a_h − a_{h−1})² / [(n−1)·Σ(a_h − ā)²]

    差分同样按环形计算（含 a[0] − a[23]），与 M10/L5 的环形口径保持一致。

    全天活动恒定时分母为 0（含全零）→ 返回 NaN：没有波动可言，
    "碎片化程度"这个量本身无定义。
    """
    arr = _validate_hourly(hourly_counts)
    n = HOURS_PER_DAY

    mean = arr.mean()
    denom = np.sum((arr - mean) ** 2)
    if denom <= 0:
        return float("nan")

    # 环形差分：np.roll 让 a[0] 与 a[23] 也构成一对相邻小时
    diffs = arr - np.roll(arr, 1)
    numer = np.sum(diffs ** 2)

    return float((n * numer) / ((n - 1) * denom))


# 已删除（2026-08-02）：compute_interdaily_stability（IS 日间稳定性指标）。
# 全仓零生产调用方——docstring 声称"只进周报层"，但 src/report/weekly_report.py
# 从不 import/调用它。留着只是"一个测了但没接进任何链路的公式"。
# 若未来周报要做节律一致性分析，可按同一公式从 compute_circadian_features 的
# 多日窗口接回来，而不是保留一份"声称被消费却没人消费"的代码。


def compute_circadian_features(hourly_counts) -> dict:
    """
    一次性算出社交轨需要的两个节律特征 + 诊断信息。

    Returns:
        {
            "rar_amplitude": float,   # 入模
            "rar_iv": float,          # 入模
            "m10": float,             # 诊断
            "l5": float,              # 诊断
            "m10_start_hour": int,    # 周报：最活跃时段
            "l5_start_hour": int,     # 周报：最安静时段
            "total_counts": float,    # 诊断：全天事件总数
        }
    """
    arr = _validate_hourly(hourly_counts)
    m10, l5, m10_start, l5_start = compute_m10_l5(arr)

    return {
        "rar_amplitude": compute_rar_amplitude(arr),
        "rar_iv": compute_rar_iv(arr),
        "m10": m10,
        "l5": l5,
        "m10_start_hour": m10_start,
        "l5_start_hour": l5_start,
        "total_counts": float(arr.sum()),
    }


def build_hourly_counts(
    event_timestamps: list[str] | None,
    dedup_window_sec: int = 60,
) -> np.ndarray:
    """
    把活动事件时间戳列表聚合成小时序列 a[0..23]。

    Args:
        event_timestamps: ISO 格式时间戳列表（"YYYY-MM-DDTHH:MM:SS"），
                          T1C PIR 与 C6c 移动/人形事件混合传入
        dedup_window_sec: 去重窗口，窗口内的多次触发合并为一次（默认 60 s）

    Returns:
        (24,) 每小时的去重后事件数

    去重按事件发生顺序做，跨小时边界的事件归入其自身所在小时——不做跨小时合并，
    否则 a[h] 的总和会与 activity_counts 不一致。
    """
    counts = np.zeros(HOURS_PER_DAY, dtype=np.float64)
    if not event_timestamps:
        return counts

    from datetime import datetime

    parsed = []
    for ts in event_timestamps:
        try:
            parsed.append(datetime.fromisoformat(ts))
        except (ValueError, TypeError):
            continue  # 脏时间戳跳过，不让整天的聚合失败

    parsed.sort()

    last_kept: dict[int, "datetime"] = {}
    for dt in parsed:
        hour = dt.hour
        prev = last_kept.get(hour)
        if prev is not None and (dt - prev).total_seconds() < dedup_window_sec:
            continue
        last_kept[hour] = dt
        counts[hour] += 1

    return counts

"""
数据聚合器：传感器原始数据 → 双轨特征向量

    睡眠轨（8维）← 小贝壳无感睡眠监测仪
        sleep_efficiency / waso_min / sol_min / bed_exit_count
        deep_sleep_ratio / sleep_onset_clock / night_hr_mean / daytime_nap_min

    社会连接轨（5维）← 萤石 C6c 摄像机 + T1C 人体移动传感器
        copresence_min / out_of_home_min / rar_amplitude / rar_iv / activity_counts

缺失判定按轨独立：小贝壳掉线不该让社交轨也停摆。
"""

import numpy as np

from src.baseline.scaler_utils import (
    SLEEP_FEATURES,
    SOCIAL_FEATURES,
    TRACK_SLEEP,
    TRACK_SOCIAL,
    get_feature_dim,
    get_feature_names,
    validate_track,
)


# 某轨缺 ≥ 此数目的特征即整轨作废
MISSING_THRESHOLD = 3

# copresence_min 禁止前向填充。其它特征反映老人自身的行为习惯，昨天的值对今天
# 有预测力；而"今天有没有人来"取决于子女的安排，用昨天填今天等于凭空伪造社会接触。
NO_FORWARD_FILL_FEATURES = frozenset({"copresence_min"})


class DataInsufficientError(Exception):
    """某轨当日数据不足以聚合（≥MISSING_THRESHOLD 个特征缺失）"""

    def __init__(self, track: str, missing_count: int, missing_features: list[str]):
        self.track = track
        self.missing_count = missing_count
        self.missing_features = missing_features
        super().__init__(
            f"Track {track!r} data insufficient: {missing_count} features missing: "
            f"{missing_features}"
        )


def _collect(raw: dict | None, names: list[str]) -> dict:
    """从原始数据字典里按名字取值，缺失或非数值一律记 None。

    显式排除 bool：Python 里 isinstance(True, int) 为真，若不排除，
    上游误传布尔标志会被静默当成 1.0/0.0 混进特征向量。
    """
    features: dict[str, float | None] = {name: None for name in names}
    if raw is None:
        return features

    for name in names:
        value = raw.get(name)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if np.isfinite(float(value)):
                features[name] = float(value)
    return features


def aggregate_sleep_features(sleep_data: dict | None) -> dict:
    """
    从小贝壳睡眠数据提取 8 维睡眠特征。

    夜间窗口以在床段动态确定（不用固定钟点），派生规则见 5.1：
        搜索区间   = [D−1 18:00, D 14:00]   ← 固定日历区间，不依赖入睡时刻
        T_bed      = 搜索区间内首个在床段的起点
        T_onset    = T_bed 之后首个 stage ∈ {LIGHT, MIDDLE, DEEP} 的时刻
        T_rise     = 该夜最后一个在床段的终点（其后离床持续 >30 min 才算真起床）
        TIB        = T_rise − T_bed
        TST        = Σ dur(stage ∈ {LIGHT, MIDDLE, DEEP})
        WASO       = Σ dur(stage == WAKEFUL) in [T_onset, T_rise]

    本函数只做"取值 + 缺失标记"，时间窗派生由适配器
    （src/data_pipeline/adapters/xiaobeike.py）完成后以扁平字典传入。

    Args:
        sleep_data: 已派生的睡眠特征字典，键为 SLEEP_FEATURES 中的名字

    Returns:
        {feature_name: value | None}
    """
    return _collect(sleep_data, SLEEP_FEATURES)


def aggregate_social_features(
    activity_data: dict | None = None,
    camera_data: dict | None = None,
) -> dict:
    """
    从 T1C PIR 事件 + C6c 摄像头事件提取 5 维社会连接特征。

    日间窗口 = [T_rise, 当晚 T_bed]。小时活动序列 a[0..23] 由两路事件去重合并
    构成（60 s 内合并为一次），rar_amplitude / rar_iv 由 a[h] 派生
    （见 adapters/circadian.py）。

    Args:
        activity_data: T1C + C6c 派生的活动类特征
                       （activity_counts / out_of_home_min / rar_amplitude / rar_iv）
        camera_data:   边缘人形检测派生的 copresence_min

    Returns:
        {feature_name: value | None}

    两路都可能只带部分特征，后到的非 None 值优先——这样 camera_data 里的
    copresence_min 不会被 activity_data 里缺失的同名键覆盖成 None。
    """
    merged = _collect(activity_data, SOCIAL_FEATURES)
    from_camera = _collect(camera_data, SOCIAL_FEATURES)
    for name, value in from_camera.items():
        if value is not None:
            merged[name] = value
    return merged


def aggregate_track_features(track: str, feature_values: dict) -> np.ndarray:
    """
    把某轨的 {特征名: 值} 字典组装成向量，缺失位填 NaN。

    Args:
        track: "sleep" / "social"
        feature_values: 该轨的特征字典，缺失项为 None

    Returns:
        (feature_dim,) 向量，缺失位为 NaN（待 imputer 填充）

    Raises:
        DataInsufficientError: 该轨缺失 ≥ MISSING_THRESHOLD 个特征
    """
    names = get_feature_names(track)

    missing_features = [n for n in names if feature_values.get(n) is None]
    missing_count = len(missing_features)

    if missing_count >= MISSING_THRESHOLD:
        raise DataInsufficientError(track, missing_count, missing_features)

    vector = np.full(get_feature_dim(track), np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        value = feature_values.get(name)
        if value is not None:
            vector[i] = float(value)

    return vector


def aggregate_daily_features(
    day_key: str,
    sleep_data: dict | None = None,
    activity_data: dict | None = None,
    camera_data: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    原始数据 → 双轨特征向量。两轨独立聚合，一轨失败不影响另一轨。

    Args:
        day_key: 自然日 "YYYY-MM-DD"（保留参数以标识数据归属日，不参与计算）
        sleep_data: 小贝壳派生特征
        activity_data: T1C + C6c 派生特征
        camera_data: 边缘人形检测派生特征

    Returns:
        {"sleep": (8,) 或 None, "social": (5,) 或 None}
        某轨数据不足时该轨为 None（不抛异常，让另一轨继续）。
        两轨都不足时才抛 DataInsufficientError。

    Raises:
        DataInsufficientError: 两轨同时数据不足（携带 sleep 轨的缺失信息）
    """
    result: dict[str, np.ndarray | None] = {TRACK_SLEEP: None, TRACK_SOCIAL: None}
    errors: dict[str, DataInsufficientError] = {}

    try:
        result[TRACK_SLEEP] = aggregate_track_features(
            TRACK_SLEEP, aggregate_sleep_features(sleep_data)
        )
    except DataInsufficientError as e:
        errors[TRACK_SLEEP] = e

    try:
        result[TRACK_SOCIAL] = aggregate_track_features(
            TRACK_SOCIAL, aggregate_social_features(activity_data, camera_data)
        )
    except DataInsufficientError as e:
        errors[TRACK_SOCIAL] = e

    # 两轨同时失败才算整体失败——单轨失败是正常的降级场景（设备掉线），
    # 上层按轨各自标 quality 即可。
    if len(errors) == len(result):
        raise errors[TRACK_SLEEP]

    return result


def diagnose_social_failure(
    feature_values: dict,
    xiaobeike_online: bool = True,
) -> dict:
    """
    社交轨的成组失效诊断。

    5 维并非相互独立：rar_amplitude / rar_iv / activity_counts 三者同源于小时活动
    序列 a[h]。因此设备故障不是"随机掉几维"，而是成组失效：

        C6c 离线      → copresence_min 无法计算（且不可填充）；a[h] 退化为仅 PIR，
                        activity_counts 系统性偏低 → 整轨 missing
        T1C 离线/欠压  → a[h] 退化为仅摄像头，覆盖范围缩小；
                        out_of_home_min 失去门厅确认 → 整轨 degraded
        小贝壳离线     → out_of_home_min 失去"非在床"条件，午睡会被误算成离家
                        → 该维标 NaN（缺 1 维），整轨仍 valid
        a[h] 全缺     → 3 维同时失效 → 触发 ≥3 维规则，整轨 missing

    Args:
        feature_values: 社交轨的 {特征名: 值|None}
        xiaobeike_online: 小贝壳是否在线（影响 out_of_home_min 的可信度）

    Returns:
        {"suspected_quality": "valid"|"degraded"|"missing", "reasons": [...]}
    """
    reasons: list[str] = []

    copresence_missing = feature_values.get("copresence_min") is None
    ah_derived = ["rar_amplitude", "rar_iv", "activity_counts"]
    ah_missing = [n for n in ah_derived if feature_values.get(n) is None]

    if copresence_missing:
        reasons.append("copresence_min 缺失（疑似 C6c 离线；该维禁止前向填充）")
    if len(ah_missing) == len(ah_derived):
        reasons.append("小时活动序列 a[h] 全缺，3 个派生维同时失效")
    elif ah_missing:
        reasons.append(f"a[h] 派生维部分缺失: {ah_missing}")
    if not xiaobeike_online and feature_values.get("out_of_home_min") is not None:
        reasons.append("小贝壳离线，out_of_home_min 失去『非在床』条件，可信度下降")

    if copresence_missing or len(ah_missing) == len(ah_derived):
        quality = "missing"
    elif ah_missing or not xiaobeike_online:
        quality = "degraded"
    else:
        quality = "valid"

    return {"suspected_quality": quality, "reasons": reasons}


def get_feature_value(vector: np.ndarray, feature_name: str, track: str) -> float:
    """从某轨特征向量中提取指定特征值"""
    names = get_feature_names(validate_track(track))
    if feature_name not in names:
        raise ValueError(f"Feature {feature_name!r} not in track {track!r}")
    return float(vector[names.index(feature_name)])

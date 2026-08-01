"""
双轨特征定义 + Scaler管理工具

两条独立的个人基线轨，各自维护自己的特征表、StandardScaler 与 GRU：

    睡眠轨（8维，主源：小贝壳无感睡眠监测仪）
    社会连接轨（5维，主源：萤石 C6c 摄像机 + T1C 人体移动传感器）

为什么拆两轨（设计文档 4.1）：
    1. 信号不被稀释——单轨把两类残差平均成一个分，睡眠恶化会被社交正常拉平
    2. 风险类型天然可解释——每轨的偏离分就是该维度的分，不需要事后猜
    3. 故障隔离——小贝壳掉线时社交轨照常工作，反之亦然
    4. 抑制过拟合——两个小模型比一个 13 维大模型稳

Scaler 归一化基准训练后不再重新拟合（防止数据漂移带来的隐藏误差），每轨各自冻结。
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# ========== 轨道标识 ==========

TRACK_SLEEP = "sleep"
TRACK_SOCIAL = "social"
TRACKS = (TRACK_SLEEP, TRACK_SOCIAL)


# ========== 特征表 ==========
# 顺序即向量下标顺序，与 config/feature_weights.json 中的顺序必须完全一致。
# 改动顺序会让已训练的模型/scaler/残差统计全部错位，必须删除基线重训。

SLEEP_FEATURES = [
    "sleep_efficiency",     # 睡眠效率 TST/TIB
    "waso_min",             # 入睡后清醒时长（分钟）
    "sol_min",              # 入睡潜伏期（分钟）
    "bed_exit_count",       # 夜间离床次数
    "deep_sleep_ratio",     # 深睡占比
    "sleep_onset_clock",    # 入睡时刻（距 20:00 的分钟偏移）
    "night_hr_mean",        # 夜间平均心率（bpm）
    "daytime_nap_min",      # 日间小睡时长（分钟）
]

SOCIAL_FEATURES = [
    "copresence_min",       # 人形共处时长（画面中人形数≥2 的累计分钟）
    "out_of_home_min",      # 疑似离家时长（分钟）
    "rar_amplitude",        # 昼夜节律相对振幅 RA = (M10−L5)/(M10+L5)
    "rar_iv",               # 昼夜节律日内变异性 IV
    "activity_counts",      # 日间活动事件总数
]

_TRACK_FEATURES: dict[str, list[str]] = {
    TRACK_SLEEP: SLEEP_FEATURES,
    TRACK_SOCIAL: SOCIAL_FEATURES,
}

SLEEP_FEATURE_DIM = len(SLEEP_FEATURES)    # 8
SOCIAL_FEATURE_DIM = len(SOCIAL_FEATURES)  # 5


def validate_track(track: str) -> str:
    """校验轨道名，非法值立即报错而不是静默返回空表。

    这道校验是有意的：轨道名打错（比如 "sleeping"）如果静默返回空特征表，
    下游会得到 0 维向量并一路跑到模型里才炸，堆栈离错误源很远。
    """
    if track not in _TRACK_FEATURES:
        raise ValueError(
            f"Unknown track: {track!r}. Expected one of {list(_TRACK_FEATURES)}"
        )
    return track


def get_feature_names(track: str) -> list[str]:
    """取某轨的特征名列表（返回副本，防止调用方误改全局表）"""
    return list(_TRACK_FEATURES[validate_track(track)])


def get_feature_dim(track: str) -> int:
    """取某轨的特征维度：sleep=8, social=5"""
    return len(_TRACK_FEATURES[validate_track(track)])


# 已删除（2026-07-31）：get_feature_index() / find_track_of_feature()。
# 两者零调用方、零测试。find_track_of_feature 的 docstring 写着"跨轨规则
# （作息节律紊乱）需要它"，但 rules.py 的规则定义里轨名是直接写在
# (track, feature, direction) 三元组里的，从来没走过反查。


# ========== Scaler ==========

# 已删除（2026-07-31）：create_scaler()。零调用方——trainer.py 直接
# `fit_scaler(StandardScaler(), data, track)`，绕过了这个 3 行包装。


def fit_scaler(scaler: StandardScaler, data: np.ndarray, track: str) -> StandardScaler:
    """
    在某轨数据上拟合scaler。

    Args:
        scaler: StandardScaler实例
        data: (n_samples, feature_dim) 特征矩阵
        track: "sleep" / "social"

    Returns:
        拟合后的scaler
    """
    expected = get_feature_dim(track)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != expected:
        raise ValueError(
            f"Track {track!r} expects {expected} features, got {data.shape[1]}"
        )
    return scaler.fit(data)


def transform_data(scaler: StandardScaler, data: np.ndarray, track: str) -> np.ndarray:
    """
    使用已有scaler归一化数据（不重新拟合）。

    关键：推理阶段只能 transform、绝不能再 fit。GRU 是在训练期的归一化尺度上学到"正常态"的，
    若对新数据重新 fit，归一化基准会随数据漂移，残差量纲跟着变，异常检测直接失真——
    今天的偏离可能因为 scaler 被"带偏"而看起来正常。基准必须冻结在建档期。

    Args:
        scaler: 已拟合的StandardScaler
        data: (n_samples, feature_dim) 或 (feature_dim,) 特征
        track: "sleep" / "social"

    Returns:
        归一化后的数据，保持输入维度
    """
    expected = get_feature_dim(track)
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(1, -1)
    if data.shape[1] != expected:
        raise ValueError(
            f"Track {track!r} expects {expected} features, got {data.shape[1]}"
        )
    result = scaler.transform(data)
    return result.flatten() if was_1d else result


# 已删除（2026-08-02）：inverse_transform。全仓零调用方——本系统从不把预测值
# 转回原始尺度（残差在归一化空间里算，方向与幅度都对归一化残差有意义）。


def save_scaler(scaler: StandardScaler, filepath: str | Path) -> None:
    """保存scaler到文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: str | Path) -> StandardScaler:
    """从文件加载scaler"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")
    return joblib.load(filepath)


# 已删除（2026-07-31）：get_scaler_stats() / check_scaler_fitted()。
# 零调用方、零测试；调试时 sklearn 的 scaler 对象本身就带 mean_ / scale_，
# 不需要包一层。

"""
全量累积EWMA基线（指数加权移动平均）

替代固定60天窗口，在50天比赛周期内也能稳定刻画长期趋势。
alpha=0.05 时，约80%权重来自最近30天。

双轨后共 3 个池（见 TrackEWMAPools）：
    睡眠轨 1 个        —— 睡眠行为的周内效应弱，不分池
    社交轨 2 个        —— 工作日 / 周末分池

为什么社交轨要分池：老人社交有强周末效应（子女周末探访）。单池会把
"周一比周日社交少"当成偏离，每个周一都误报一次。
"""

import pickle
from pathlib import Path


class CumulativeEWMABaseline:
    """
    全量累积基线：指数加权移动平均。

    Attributes:
        alpha: 指数衰减因子（0 < alpha < 1）
        mean: 当前EWMA均值
        m2: 用于计算运行标准差的中间量
        n: 已更新的样本数
        history: 所有历史值列表（用于回溯分析）

    Usage:
        >>> ewma = CumulativeEWMABaseline(alpha=0.05)
        >>> ewma.update(1.2)
        >>> ewma.update(0.8)
        >>> threshold = ewma.get_threshold(2.5)
    """

    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.mean: float | None = None
        self.m2: float = 0.0
        self.n: int = 0
        self.history: list[float] = []

    def update(self, new_value: float) -> None:
        """
        增量更新EWMA均值和运行方差。

        使用标准的指数加权移动方差公式：
        var_ewma = alpha * (x - mean_old)^2 + (1 - alpha) * var_ewma

        Args:
            new_value: 新的异常分值
        """
        if not isinstance(new_value, (int, float)):
            raise TypeError(f"new_value must be numeric, got {type(new_value)}")

        self.history.append(new_value)

        if self.mean is None:
            # 第一个值：无历史可平滑，直接以它为初始均值；方差此刻无从估计，置 0。
            # 这会让最初一两天的阈值偏窄，但冷启动观察期本就不报警，等样本累积后自然收敛。
            self.mean = float(new_value)
            self.m2 = 0.0
            self.n = 1
        else:
            old_mean = self.mean
            # 更新 EWMA 均值：新值权重 alpha，历史权重 (1-alpha)。alpha 越小越"记性长"。
            self.mean = (1 - self.alpha) * self.mean + self.alpha * float(new_value)
            self.n += 1

            # EWMA 方差递推：var = alpha * (x - old_mean)^2 + (1 - alpha) * var
            # 注意用的是 old_mean（更新前的均值），这是 EWMV 的常用近似式——不是无偏样本方差，
            # 但对"动态阈值"这个用途足够：它同样以指数衰减跟踪波动幅度，且 O(1) 增量可算，
            # 无需保留全部历史。m2 直接存方差本身（非 Welford 的平方和），故下方开方即得 std。
            delta_squared = (float(new_value) - old_mean) ** 2
            self.m2 = self.alpha * delta_squared + (1 - self.alpha) * self.m2

    @property
    def std(self) -> float:
        """
        EWMA运行标准差。

        Returns:
            标准差，n<2时返回1.0作为默认值
        """
        # 只有 1 个样本时方差无意义，返回 1.0 而非 0：若返回 0，get_threshold 会退化成
        # threshold == mean，任何微小波动都越阈误报。用 1.0 这个中性尺度先"撑住"阈值宽度。
        if self.n < 2:
            return 1.0
        # m2 存储的就是方差，直接开方；下限 1e-8 防止方差塌成 0 时阈值失去宽度。
        return max(self.m2 ** 0.5, 1e-8)

    @property
    def variance(self) -> float:
        """EWMA运行方差"""
        return self.std ** 2

    def get_threshold(self, sigma_multiplier: float = 2.5) -> float:
        """
        计算动态阈值：mean + sigma_multiplier * std。

        Args:
            sigma_multiplier: σ倍数，默认2.5

        Returns:
            动态阈值
        """
        return self.mean + sigma_multiplier * self.std

    def get_percentile(self, percentile: float) -> float:
        """
        基于历史数据的经验百分位数。

        Args:
            percentile: 百分位 (0-100)

        Returns:
            对应百分位的值
        """
        if len(self.history) < 3:
            return self.get_threshold(2.5)
        import numpy as np
        return float(np.percentile(self.history, percentile))

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "alpha": self.alpha,
            "mean": self.mean,
            "m2": self.m2,
            "n": self.n,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CumulativeEWMABaseline":
        """从字典反序列化"""
        ewma = cls(alpha=data["alpha"])
        ewma.mean = data["mean"]
        ewma.m2 = data["m2"]
        ewma.n = data["n"]
        ewma.history = data.get("history", [])
        return ewma

    def save(self, filepath: str | Path) -> None:
        """保存EWMA到文件（pickle格式）"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filepath: str | Path) -> "CumulativeEWMABaseline":
        """从文件加载EWMA"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"EWMA file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"CumulativeEWMABaseline(alpha={self.alpha}, n={self.n}, "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )

    def reset(self) -> None:
        """重置EWMA状态（用于重新初始化）"""
        self.mean = None
        self.m2 = 0.0
        self.n = 0
        self.history = []


# ===== 双轨池容器 =====

POOL_DEFAULT = "default"    # 睡眠轨用（不分池）
POOL_WEEKDAY = "weekday"    # 社交轨工作日池
POOL_WEEKEND = "weekend"    # 社交轨周末池


def get_pool_name(track: str, is_weekend: bool) -> str:
    """某轨某天该用哪个池。睡眠轨恒为 default，社交轨按是否周末分。"""
    if track == "social":
        return POOL_WEEKEND if is_weekend else POOL_WEEKDAY
    return POOL_DEFAULT


class TrackEWMAPools:
    """
    某一轨的 EWMA 池集合。

    睡眠轨只有 {default}；社交轨有 {weekday, weekend}。
    每个池是一个独立的 CumulativeEWMABaseline —— 类本身不做任何改动，
    分池逻辑完全在容器层，这样 EWMA 的算法（含"取 min"动态阈值策略）保持原样。

    Usage:
        >>> pools = TrackEWMAPools("social", alpha=0.05)
        >>> pools.update(1.2, is_weekend=False)
        >>> pools.get_threshold(2.5, is_weekend=False)
    """

    def __init__(self, track: str, alpha: float = 0.05):
        from src.baseline.scaler_utils import validate_track

        self.track = validate_track(track)
        self.alpha = alpha
        pool_names = (
            (POOL_WEEKDAY, POOL_WEEKEND) if self.track == "social" else (POOL_DEFAULT,)
        )
        self.pools: dict[str, CumulativeEWMABaseline] = {
            name: CumulativeEWMABaseline(alpha=alpha) for name in pool_names
        }

    def pool_for(self, is_weekend: bool = False) -> CumulativeEWMABaseline:
        """取该天对应的池"""
        return self.pools[get_pool_name(self.track, is_weekend)]

    def update(self, value: float, is_weekend: bool = False) -> None:
        """把今天的异常分喂给对应的池"""
        self.pool_for(is_weekend).update(value)

    def get_threshold(self, sigma_multiplier: float, is_weekend: bool = False) -> float:
        """对应池的动态阈值"""
        return self.pool_for(is_weekend).get_threshold(sigma_multiplier)

    def n_samples(self, is_weekend: bool = False) -> int:
        """对应池已累积的样本数（用于判断动态阈值是否够稳）"""
        return self.pool_for(is_weekend).n

    def min_samples_required(self, config: dict, is_weekend: bool = False) -> int:
        """
        该池启用动态阈值所需的最小样本数。

        周末池的门槛单独设（默认 8 而非 20）：周末只占 2/7，攒 20 个要 70 天。
        ⚠️ 即便降到 8 也需约 8 个周末≈56 天，所以社交轨的周末动态阈值到第 8 周
        才真正上线，此前只有静态阈值。这是样本稀疏的算术后果，调参消不掉。
        """
        ewma_cfg = config.get("ewma", {})
        if self.track == "social" and is_weekend:
            return ewma_cfg.get("min_samples_for_dynamic_weekend", 8)
        return ewma_cfg.get("min_samples_for_dynamic", 20)

    def total_samples(self) -> int:
        """全部池的样本数之和（观察期判定用）"""
        return sum(p.n for p in self.pools.values())

    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "alpha": self.alpha,
            "pools": {name: pool.to_dict() for name, pool in self.pools.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackEWMAPools":
        obj = cls(track=data["track"], alpha=data["alpha"])
        for name, pool_data in data.get("pools", {}).items():
            if name in obj.pools:
                obj.pools[name] = CumulativeEWMABaseline.from_dict(pool_data)
        return obj

    def save(self, dirpath: str | Path) -> None:
        """
        保存到目录：每个池一个 pkl 文件。

        分文件而不是一个大 pkl：这样某个池的文件损坏时，另一个池仍可加载，
        且文件名（ewma_social_weekend.pkl）直接说明里面是什么。
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        for name, pool in self.pools.items():
            pool.save(dirpath / self._pool_filename(name))

    def _pool_filename(self, pool_name: str) -> str:
        if pool_name == POOL_DEFAULT:
            return f"ewma_{self.track}.pkl"
        return f"ewma_{self.track}_{pool_name}.pkl"

    @classmethod
    def load(cls, dirpath: str | Path, track: str, alpha: float = 0.05) -> "TrackEWMAPools":
        """
        从目录加载。缺失的池以全新实例补位（不抛异常）——某个池还没攒到数据
        是正常状态（比如建档期恰好没跨过周末）。
        """
        dirpath = Path(dirpath)
        obj = cls(track=track, alpha=alpha)
        for name in list(obj.pools):
            filepath = dirpath / obj._pool_filename(name)
            if filepath.exists():
                obj.pools[name] = CumulativeEWMABaseline.load(filepath)
        return obj

    def __repr__(self) -> str:
        detail = ", ".join(
            f"{name}(n={pool.n})" for name, pool in self.pools.items()
        )
        return f"TrackEWMAPools(track={self.track}, {detail})"

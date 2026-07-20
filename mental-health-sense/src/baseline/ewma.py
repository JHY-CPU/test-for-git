"""
全量累积EWMA基线（指数加权移动平均）

替代固定60天窗口，在50天比赛周期内也能稳定刻画长期趋势。
alpha=0.05 时，约80%权重来自最近30天。
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
            # 第一个值：初始化
            self.mean = float(new_value)
            self.m2 = 0.0  # 初始方差为0
            self.n = 1
        else:
            old_mean = self.mean
            # 更新 EWMA 均值
            self.mean = (1 - self.alpha) * self.mean + self.alpha * float(new_value)
            self.n += 1

            # 更新 EWMA 方差：var = alpha * (x - old_mean)^2 + (1 - alpha) * var
            # 由于 m2 存储的是方差，直接使用标准公式
            delta_squared = (float(new_value) - old_mean) ** 2
            self.m2 = self.alpha * delta_squared + (1 - self.alpha) * self.m2

    @property
    def std(self) -> float:
        """
        EWMA运行标准差。

        Returns:
            标准差，n<2时返回1.0作为默认值
        """
        if self.n < 2:
            return 1.0
        # m2 存储的就是方差，直接开方
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

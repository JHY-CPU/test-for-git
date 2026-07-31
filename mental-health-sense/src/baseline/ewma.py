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

        n=0（一个样本都没喂过）时 mean 是 None，直接相加会抛 TypeError。
        生产路径被 `pool_n >= min_samples`（≥8）挡住，但 get_percentile 与
        测试/排查代码会踩到。返回 inf 而不是 0：阈值的语义是"超过它才算偏离"，
        没有基线时应当**永不触发**，返回 0 会让任何分数都判偏离。
        """
        if self.mean is None:
            return float("inf")
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
        from src.utils.io import atomic_write_bytes
        atomic_write_bytes(filepath, pickle.dumps(self.to_dict()))

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
        mean_text = "None" if self.mean is None else f"{self.mean:.4f}"
        return (
            f"CumulativeEWMABaseline(alpha={self.alpha}, n={self.n}, "
            f"mean={mean_text}, std={self.std:.4f})"
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

    def __init__(self, track: str, alpha: float = 0.05, max_freeze_days: int = 14):
        from src.baseline.scaler_utils import validate_track

        self.track = validate_track(track)
        self.alpha = alpha
        pool_names = (
            (POOL_WEEKDAY, POOL_WEEKEND) if self.track == "social" else (POOL_DEFAULT,)
        )
        self.pools: dict[str, CumulativeEWMABaseline] = {
            name: CumulativeEWMABaseline(alpha=alpha) for name in pool_names
        }
        # 冻结上限：连续偏离超过这么多天后恢复更新，避免"永久报警"
        self.max_freeze_days = max_freeze_days
        self.freeze_streak = 0
        # 已喂入的最后一个自然日。用于拒绝重复喂入，见 update()。
        self.last_day_key: str | None = None

    def pool_for(self, is_weekend: bool = False) -> CumulativeEWMABaseline:
        """取该天对应的池"""
        return self.pools[get_pool_name(self.track, is_weekend)]

    def update(
        self,
        value: float,
        is_weekend: bool = False,
        is_deviation: bool = False,
        day_key: str | None = None,
    ) -> bool:
        """
        把今天的异常分喂给对应的池。

        ★ 同一自然日只喂一次（day_key 去重）

        infer_track 每次调用都无条件 update + 立即落盘，而"补算/重跑某一天"是
        被明确预期的用法：save_daily_features 做了同日幂等覆盖，daily_inference
        的连续天数统计也专门跳过"今天自己的旧记录"。只有 EWMA 漏了这条——
        重跑一次 run_daily_pipeline --date X，那天的分就被喂进基线两次，
        freeze_streak 也跟着多加一次。

        用 `day_key <= last_day_key` 而不是等值比较：这样既挡住重跑，也挡住
        乱序补算历史日（往回补一天会把早已过去的分当成最新观测喂进指数加权，
        权重完全错位）。

        ★ 偏离日**不更新**（冻结）。理由：EWMA 是"正常波动"的基线，
        若把异常日也喂进去，基线会在两三天内学会这次异常，阈值随分数一起抬高，
        于是持续性异常被自己的历史掩盖——偏离标志开始闪烁，
        永远凑不满"连续 N 天"，等级卡在 L1。这是自适应基线的经典失效模式。

        实测（TP_social，共处归零 11 天）：不冻结时阈值 1.37→2.02 一路追平分数，
        11 天里只有 5 天被判偏离且不连续；冻结后阈值稳定在 1.37 附近。

        但不能无限冻结：老人若真的永久性衰退（搬家后再没人来），
        基线该重新学习，否则会永久报警、家属很快对提醒脱敏。
        故连续冻结超过 max_freeze_days 天后强制恢复更新——
        此时"异常"已持续两周，风险规则该报的早已报过，
        重新基线化是为了让系统对**下一次**变化仍然敏感。

        Returns:
            True 表示本次实际更新了池，False 表示被冻结或被去重跳过。
        """
        if day_key is not None and self.last_day_key is not None:
            if day_key <= self.last_day_key:
                return False

        if is_deviation and self.freeze_streak < self.max_freeze_days:
            self.freeze_streak += 1
            if day_key is not None:
                # 冻结日也要推进游标：否则重跑冻结日会让 freeze_streak 反复自增，
                # 14 天上限提前触发、基线过早重新学习这次异常。
                self.last_day_key = day_key
            return False

        if not is_deviation:
            self.freeze_streak = 0
        self.pool_for(is_weekend).update(value)
        if day_key is not None:
            self.last_day_key = day_key
        return True

    def already_fed(self, day_key: str | None) -> bool:
        """该自然日是否已经被喂过（即本次是补算/重跑）。

        ★ 为什么调用方需要在 update **之前**知道这件事

          update 的去重只挡住了"再喂一次"，但阈值是在 update **之前**从池里读的。
          首跑时读到的是"不含今天"的池，落盘后池里就有今天了；重跑时读到的池
          已经包含这一天自己的分，于是拿"含被判对象的基线"去判这个对象。

          实测影响：某非偏离日 score=1.28、阈值 1.30 → 不偏离，池被更新
          （mean 上移 alpha·(1.28−mean)，m2 也跟着变）。重跑同一天时
          mean + 2.5·std 已经移动，落在新旧阈值之间的分数会**翻转 is_deviation**，
          而这个字段驱动 consecutive、risk_type_qualifies、微调排除集、周报统计。

          README 明写"补算与重跑是安全的"，那就必须真的幂等。
        """
        return (
            day_key is not None
            and self.last_day_key is not None
            and day_key <= self.last_day_key
        )

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
            "max_freeze_days": self.max_freeze_days,
            "freeze_streak": self.freeze_streak,
            "last_day_key": self.last_day_key,
            "pools": {name: pool.to_dict() for name, pool in self.pools.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackEWMAPools":
        obj = cls(
            track=data["track"],
            alpha=data["alpha"],
            max_freeze_days=data.get("max_freeze_days", 14),
        )
        # freeze_streak 必须持久化：它跨天累积，每天是独立进程，
        # 不落盘的话每天都从 0 开始，冻结上限永远触发不到。
        obj.freeze_streak = data.get("freeze_streak", 0)
        obj.last_day_key = data.get("last_day_key")
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

        # 容器级状态（冻结计数）单独落一个小文件：池文件的格式不动，
        # 老基线目录缺这个文件时按 0 起算，向后兼容。
        from src.utils.io import atomic_write_bytes
        atomic_write_bytes(
            dirpath / self._state_filename(),
            pickle.dumps({
                "freeze_streak": self.freeze_streak,
                "max_freeze_days": self.max_freeze_days,
                "last_day_key": self.last_day_key,
            }),
        )

    def _pool_filename(self, pool_name: str) -> str:
        if pool_name == POOL_DEFAULT:
            return f"ewma_{self.track}.pkl"
        return f"ewma_{self.track}_{pool_name}.pkl"

    def _state_filename(self) -> str:
        return f"ewma_{self.track}_state.pkl"

    @classmethod
    def load(
        cls,
        dirpath: str | Path,
        track: str,
        alpha: float = 0.05,
        max_freeze_days: int = 14,
    ) -> "TrackEWMAPools":
        """
        从目录加载。缺失的池以全新实例补位（不抛异常）——某个池还没攒到数据
        是正常状态（比如建档期恰好没跨过周末）。

        max_freeze_days 由调用方从配置传入；状态文件里存的值优先（见下方），
        这样改配置对新建档生效，已有基线保持自己建档时的口径直到重新建档。
        """
        dirpath = Path(dirpath)
        obj = cls(track=track, alpha=alpha, max_freeze_days=max_freeze_days)
        for name in list(obj.pools):
            filepath = dirpath / obj._pool_filename(name)
            if filepath.exists():
                obj.pools[name] = CumulativeEWMABaseline.load(filepath)

        state_path = dirpath / obj._state_filename()
        if state_path.exists():
            try:
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                obj.freeze_streak = int(state.get("freeze_streak", 0))
                obj.max_freeze_days = int(
                    state.get("max_freeze_days", obj.max_freeze_days)
                )
                obj.last_day_key = state.get("last_day_key")
            except Exception:
                # 状态文件坏了不该让整轨加载失败：冻结计数丢失只是让阈值
                # 早一天恢复更新，比拿不到基线严重得多。
                obj.freeze_streak = 0
                obj.last_day_key = None
        return obj

    def __repr__(self) -> str:
        detail = ", ".join(
            f"{name}(n={pool.n})" for name, pool in self.pools.items()
        )
        return f"TrackEWMAPools(track={self.track}, {detail})"

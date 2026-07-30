"""_train_loop 早停/回滚回归测试（待修清单 P2-2：微调与冷启动共用同一策略）。"""

import torch

from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.trainer import _train_loop


DIM = 8  # 睡眠轨维度


def _fixture(dropout=0.2):
    torch.manual_seed(0)
    model = PersonalBaselineGRU(feature_dim=DIM, hidden_dim=8, num_layers=1, dropout=dropout)
    X = torch.randn(6, 7, DIM)
    y = torch.randn(6, DIM)
    return model, X, y


class TestTrainLoop:
    def test_returns_finite_best_loss(self):
        model, X, y = _fixture()
        best = _train_loop(model, X, y, epochs=30, lr=0.01, patience=5)
        assert best < float("inf")
        assert best >= 0.0

    def test_early_stopping_rolls_back_to_best(self):
        """early-stopping 后模型权重应等于最优点权重（best_loss 可复现）。

        用 dropout=0 消除训练态随机性，train/eval 前向一致，才能拿 eval 损失
        校验"回滚到最优权重"这件事本身。
        """
        model, X, y = _fixture(dropout=0.0)
        best = _train_loop(model, X, y, epochs=100, lr=0.05, patience=3)
        # 回滚后，用当前权重重算训练损失应 == best_loss（±数值误差）
        model.eval()
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(model(X), y).item()
        assert abs(loss - best) < 1e-4

    def test_no_patience_runs_full(self):
        model, X, y = _fixture()
        best = _train_loop(model, X, y, epochs=20, lr=0.01, patience=None)
        assert best < float("inf")


class TestWindowSplit:
    """训练段/留出段切分：留出段目标日绝不能参与训练"""

    def test_window_shapes(self):
        import numpy as np
        from src.baseline.trainer import _build_windows

        data = np.arange(35 * DIM, dtype=np.float64).reshape(35, DIM)
        X, y, idx = _build_windows(data, window=7, start=7, end=28)
        assert X.shape == (21, 7, DIM)
        assert y.shape == (21, DIM)
        assert idx == list(range(7, 28))

    def test_holdout_targets_disjoint_from_train(self):
        """★ 关键不变量：两段的目标日索引不能有交集"""
        import numpy as np
        from src.baseline.trainer import _build_windows

        data = np.zeros((35, DIM), dtype=np.float64)
        _, _, train_idx = _build_windows(data, 7, 7, 28)
        _, _, hold_idx = _build_windows(data, 7, 28, 35)
        assert set(train_idx).isdisjoint(set(hold_idx))
        assert len(hold_idx) == 7

    def test_empty_range_returns_empty(self):
        import numpy as np
        from src.baseline.trainer import _build_windows

        data = np.zeros((10, DIM), dtype=np.float64)
        X, y, idx = _build_windows(data, 7, 7, 7)
        assert len(X) == 0 and len(y) == 0 and idx == []


class TestResidualStats:
    """双残差契约：signed 判方向，abs 打幅度"""

    def test_signed_is_observed_minus_predicted(self):
        """★ 符号约定：观测低于预测 → signed 为负（与"下降"直觉一致）"""
        from src.baseline.trainer import _residual_stats_from

        pred = torch.full((5, DIM), 1.0)
        target = torch.full((5, DIM), 0.4)   # 观测值低于预测
        stats = _residual_stats_from(pred, target)

        assert stats["signed"]["mean"][0] < 0, "观测低于预测应为负"
        assert abs(stats["signed"]["mean"][0] + 0.6) < 1e-6
        assert abs(stats["abs"]["mean"][0] - 0.6) < 1e-6

    def test_both_kinds_present(self):
        from src.baseline.trainer import _residual_stats_from

        stats = _residual_stats_from(torch.randn(9, DIM), torch.randn(9, DIM))
        assert set(stats) == {"signed", "abs"}
        for kind in ("signed", "abs"):
            assert set(stats[kind]) == {"mean", "std"}
            assert stats[kind]["mean"].shape == (DIM,)

    def test_signed_std_differs_from_abs_std(self):
        """量纲差异的证据：|r| 的分布与 r 不同，不能互相顶替。

        这正是旧实现的问题——存 abs 统计却拿它的 std 去标准化 signed 残差。
        """
        from src.baseline.trainer import _residual_stats_from

        torch.manual_seed(3)
        stats = _residual_stats_from(torch.zeros(200, DIM), torch.randn(200, DIM))
        assert stats["signed"]["std"][0] > stats["abs"]["std"][0]


class TestEWMAPools:
    """双轨三池：睡眠 1 个，社交按 is_weekend 分 2 个"""

    def test_sleep_has_single_pool(self):
        from src.baseline.ewma import TrackEWMAPools

        pools = TrackEWMAPools("sleep", alpha=0.05)
        assert set(pools.pools) == {"default"}
        # 睡眠轨不分池：周末与工作日落到同一个池
        pools.update(1.0, is_weekend=False)
        pools.update(2.0, is_weekend=True)
        assert pools.n_samples(is_weekend=False) == 2

    def test_social_pools_are_independent(self):
        """★ 分池的核心：周末数据不污染工作日基线"""
        from src.baseline.ewma import TrackEWMAPools

        pools = TrackEWMAPools("social", alpha=0.05)
        assert set(pools.pools) == {"weekday", "weekend"}

        for _ in range(5):
            pools.update(1.0, is_weekend=False)
        pools.update(9.0, is_weekend=True)

        assert pools.n_samples(is_weekend=False) == 5
        assert pools.n_samples(is_weekend=True) == 1
        assert pools.pool_for(False).mean < pools.pool_for(True).mean

    def test_weekend_min_samples_lower(self):
        """周末样本占 2/7，门槛必须单独降低，否则动态阈值永远不上线"""
        from src.baseline.ewma import TrackEWMAPools

        config = {"ewma": {"min_samples_for_dynamic": 20,
                           "min_samples_for_dynamic_weekend": 8}}
        pools = TrackEWMAPools("social")
        assert pools.min_samples_required(config, is_weekend=False) == 20
        assert pools.min_samples_required(config, is_weekend=True) == 8

    def test_roundtrip_save_load(self, tmp_path):
        from src.baseline.ewma import TrackEWMAPools

        pools = TrackEWMAPools("social", alpha=0.1)
        for i in range(4):
            pools.update(float(i), is_weekend=(i % 2 == 0))
        pools.save(tmp_path)

        assert (tmp_path / "ewma_social_weekday.pkl").exists()
        assert (tmp_path / "ewma_social_weekend.pkl").exists()

        loaded = TrackEWMAPools.load(tmp_path, "social", alpha=0.1)
        assert loaded.n_samples(is_weekend=False) == pools.n_samples(is_weekend=False)
        assert loaded.n_samples(is_weekend=True) == pools.n_samples(is_weekend=True)

    def test_load_missing_files_is_fresh(self, tmp_path):
        """某池还没攒到数据是正常状态（建档期恰好没跨周末），不该抛异常"""
        from src.baseline.ewma import TrackEWMAPools

        loaded = TrackEWMAPools.load(tmp_path, "sleep")
        assert loaded.total_samples() == 0

    def test_ewma_algorithm_unchanged(self):
        """回归护栏：底层 CumulativeEWMABaseline 的行为不因分池而改变"""
        from src.baseline.ewma import CumulativeEWMABaseline, TrackEWMAPools

        direct = CumulativeEWMABaseline(alpha=0.05)
        pooled = TrackEWMAPools("sleep", alpha=0.05)
        for v in (1.0, 2.0, 1.5, 3.0):
            direct.update(v)
            pooled.update(v)
        assert abs(direct.mean - pooled.pool_for().mean) < 1e-12
        assert abs(direct.std - pooled.pool_for().std) < 1e-12
        assert abs(direct.get_threshold(2.5) - pooled.get_threshold(2.5)) < 1e-12

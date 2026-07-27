"""_train_loop 早停/回滚回归测试（待修清单 P2-2：微调与冷启动共用同一策略）。"""

import torch

from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.trainer import _train_loop


def _fixture(dropout=0.2):
    torch.manual_seed(0)
    model = PersonalBaselineGRU(feature_dim=10, hidden_dim=16, num_layers=1, dropout=dropout)
    X = torch.randn(6, 7, 10)
    y = torch.randn(6, 10)
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

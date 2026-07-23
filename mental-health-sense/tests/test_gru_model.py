"""
GRU模型单元测试
"""

import numpy as np
import pytest
import torch

from src.baseline.gru_model import PersonalBaselineGRU


class TestPersonalBaselineGRU:
    """测试个人基线GRU模型（特征维度已更新为10）"""

    @pytest.fixture
    def model(self):
        return PersonalBaselineGRU(
            feature_dim=10,
            hidden_dim=16,
            num_layers=1,
        )

    @pytest.fixture
    def sample_input(self):
        # (batch=4, window=7, features=10)
        return torch.randn(4, 7, 10)

    def test_forward_shape(self, model, sample_input):
        """测试前向传播输出形状"""
        output = model(sample_input)
        assert output.shape == (4, 10)

    def test_single_batch(self, model):
        """测试单样本输入"""
        x = torch.randn(1, 7, 10)
        output = model(x)
        assert output.shape == (1, 10)

    def test_gradient_flow(self, model, sample_input):
        """测试梯度流动"""
        output = model(sample_input)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.all(param.grad == 0), f"{name} gradient is all zeros"

    def test_predict_mode(self, model, sample_input):
        """测试推理模式（无梯度）"""
        output = model.predict(sample_input)
        assert output.shape == (4, 10)
        assert not output.requires_grad

    def test_get_hidden_state(self, model, sample_input):
        """测试隐藏状态提取"""
        hidden = model.get_hidden_state(sample_input)
        assert hidden.shape == (4, 16)  # hidden_dim=16

    def test_parameter_count(self):
        """测试参数量统计（极轻量）"""
        model = PersonalBaselineGRU(
            feature_dim=10,
            hidden_dim=16,
            num_layers=1,
        )
        n_params = model.count_parameters()
        assert n_params < 2000, f"Expected <2000 params, got {n_params}"

    def test_reset_parameters(self, model):
        """测试参数重置"""
        old_weights = {}
        for name, param in model.named_parameters():
            old_weights[name] = param.data.clone()

        model.reset_parameters()

        for name, param in model.named_parameters():
            if "weight" in name:
                assert not torch.equal(param.data, old_weights[name]), \
                    f"{name} did not change after reset"

    def test_overfit_small_data(self):
        """测试在小数据集上的过拟合能力（验证模型容量）"""
        # 固定种子：本测试断言 loss<0.01，未设种子时随机初始化会让终值在
        # 阈值附近抖动（曾观测到 0.0101 偶发越线），设种子使其确定可复现。
        # seed=100 收敛到约 2e-5，留有充足余量。
        torch.manual_seed(100)
        model = PersonalBaselineGRU(feature_dim=4, hidden_dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        X = torch.tensor([[
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7],
            [0.5, 0.6, 0.7, 0.8],
            [0.6, 0.7, 0.8, 0.9],
            [0.7, 0.8, 0.9, 1.0],
        ]], dtype=torch.float32)
        y = torch.tensor([[0.8, 0.9, 1.0, 1.1]], dtype=torch.float32)

        model.train()
        for _ in range(300):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_pred = model(X)
            final_loss = loss_fn(final_pred, y)
        assert final_loss.item() < 0.01, f"Model failed to overfit: loss={final_loss.item():.4f}"

    def test_dropout_training_vs_eval(self, model, sample_input):
        """测试dropout在train/eval模式下的行为差异"""
        model.train()
        out1 = model(sample_input)
        out2 = model(sample_input)
        assert not torch.equal(out1, out2)

        model.eval()
        out3 = model(sample_input)
        out4 = model(sample_input)
        assert torch.equal(out3, out4)

    def test_invalid_params(self):
        """测试非法参数"""
        with pytest.raises(ValueError):
            PersonalBaselineGRU(feature_dim=0)
        with pytest.raises(ValueError):
            PersonalBaselineGRU(hidden_dim=0)
        with pytest.raises(ValueError):
            PersonalBaselineGRU(num_layers=0)

    def test_model_repr(self, model):
        """测试模型字符串表示"""
        rep = repr(model)
        assert "PersonalBaselineGRU" in rep
        assert "feature_dim=10" in rep
        assert "hidden_dim=16" in rep

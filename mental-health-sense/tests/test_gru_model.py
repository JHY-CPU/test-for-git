"""
GRU模型单元测试
"""

import numpy as np
import pytest
import torch

from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import SLEEP_FEATURE_DIM, SOCIAL_FEATURE_DIM


class TestPersonalBaselineGRU:
    """测试个人基线GRU模型（睡眠轨 8 维 / 社会连接轨 5 维）"""

    @pytest.fixture
    def model(self):
        return PersonalBaselineGRU(
            feature_dim=SLEEP_FEATURE_DIM,
            hidden_dim=8,
            num_layers=1,
        )

    @pytest.fixture
    def sample_input(self):
        # (batch=4, window=7, features=8)
        return torch.randn(4, 7, SLEEP_FEATURE_DIM)

    def test_forward_shape(self, model, sample_input):
        """测试前向传播输出形状"""
        output = model(sample_input)
        assert output.shape == (4, SLEEP_FEATURE_DIM)

    def test_single_batch(self, model):
        """测试单样本输入"""
        x = torch.randn(1, 7, SLEEP_FEATURE_DIM)
        output = model(x)
        assert output.shape == (1, SLEEP_FEATURE_DIM)

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
        assert output.shape == (4, SLEEP_FEATURE_DIM)
        assert not output.requires_grad

    def test_get_hidden_state(self, model, sample_input):
        """测试隐藏状态提取"""
        hidden = model.get_hidden_state(sample_input)
        assert hidden.shape == (4, 8)  # hidden_dim=8

    def test_parameter_count_sleep(self):
        """睡眠轨 8 维 + hidden 8 = 504 参数。

        ★ 这个数字是有意压低的：hidden=12 时是 896 参数，而 35 天建档期只切得出
        21 个训练序列。参数远多于样本会让模型记住序列而非学出基线，残差趋零，
        阈值分母趋零，建档期一过就疯狂误报。
        """
        model = PersonalBaselineGRU(feature_dim=SLEEP_FEATURE_DIM, hidden_dim=8)
        assert model.count_parameters() == 504

    def test_parameter_count_social(self):
        """社会连接轨 5 维 + hidden 8 = 405 参数"""
        model = PersonalBaselineGRU(feature_dim=SOCIAL_FEATURE_DIM, hidden_dim=8)
        assert model.count_parameters() == 405

    def test_params_scale_with_hidden(self):
        """回归护栏：hidden 调大会让参数量迅速膨胀，改配置时应看得见代价"""
        small = PersonalBaselineGRU(feature_dim=SLEEP_FEATURE_DIM, hidden_dim=8)
        big = PersonalBaselineGRU(feature_dim=SLEEP_FEATURE_DIM, hidden_dim=12)
        assert big.count_parameters() == 896
        assert big.count_parameters() > small.count_parameters() * 1.7

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
            PersonalBaselineGRU(feature_dim=8, hidden_dim=0)
        with pytest.raises(ValueError):
            PersonalBaselineGRU(feature_dim=8, num_layers=0)

    def test_feature_dim_is_required(self):
        """feature_dim 必须显式传入——双轨维度不同，默认值会掩盖用错轨的错误"""
        with pytest.raises(TypeError):
            PersonalBaselineGRU()

    def test_model_repr(self, model):
        """测试模型字符串表示"""
        rep = repr(model)
        assert "PersonalBaselineGRU" in rep
        assert "feature_dim=8" in rep
        assert "hidden_dim=8" in rep

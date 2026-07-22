"""
个人基线GRU模型定义

输入：过去7天的10维特征向量
输出：第8天的10维特征向量预测

系统为被监测的老人独立维护一个GRU模型，预测残差作为"偏离个人常态"的量化依据。
"""

import torch
import torch.nn as nn


class PersonalBaselineGRU(nn.Module):
    """
    个人基线GRU模型：用过去7天预测第8天。

    Architecture:
        GRU(input_dim=10, hidden_dim=16, num_layers=1)
        → Linear(16, 10)

    Input shape:  (batch, 7, 10)
    Output shape: (batch, 10)

    Args:
        feature_dim: 输入特征维度，默认10（健康特征，已移除时间编码）
        hidden_dim: GRU隐藏层维度，默认16（极轻量，防止过拟合）
        num_layers: GRU层数，默认1
        dropout: Dropout比率，默认0.2

    Usage:
        >>> model = PersonalBaselineGRU()
        >>> x = torch.randn(32, 7, 10)  # (batch, 7天, 10特征)
        >>> pred = model(x)              # (batch, 10)
    """

    def __init__(
        self,
        feature_dim: int = 10,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: (batch_size, 7, feature_dim) 过去7天的特征序列

        Returns:
            (batch_size, feature_dim) 第8天预测值
        """
        out, _ = self.gru(x)  # (batch, 7, hidden_dim)
        last_out = out[:, -1, :]  # (batch, hidden_dim)
        last_out = self.dropout(last_out)
        return self.fc(last_out)  # (batch, feature_dim)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理模式预测（自动关闭梯度计算）。

        Args:
            x: (batch_size, 7, feature_dim)

        Returns:
            (batch_size, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取GRU最后时刻的隐藏状态（用于可解释性分析）。

        Args:
            x: (batch_size, 7, feature_dim)

        Returns:
            (batch_size, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            out, _ = self.gru(x)
            return out[:, -1, :]

    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self) -> None:
        """重新初始化模型参数"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def __repr__(self) -> str:
        return (
            f"PersonalBaselineGRU("
            f"feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"params={self.count_parameters()})"
        )

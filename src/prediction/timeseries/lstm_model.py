"""Small single-layer multi-head LSTM registered by the daily-v1 contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LstmLoss:
    total: torch.Tensor
    direction: torch.Tensor
    expected_return: torch.Tensor
    interval: torch.Tensor


class SmallLstmMultiHead(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        external_dropout: float,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("LSTM input size must be positive")
        if hidden_size not in (16, 32):
            raise ValueError("LSTM v1 hidden size must be 16 or 32")
        if not 0.0 <= external_dropout < 1.0:
            raise ValueError("external dropout must be in [0, 1)")
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(external_dropout)
        self.direction_head = nn.Linear(hidden_size, 1)
        self.return_head = nn.Linear(hidden_size, 1)
        self.lower_distance_head = nn.Linear(hidden_size, 1)
        self.upper_distance_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.return_head.bias)
        nn.init.zeros_(self.lower_distance_head.weight)
        nn.init.zeros_(self.upper_distance_head.weight)
        nn.init.constant_(self.lower_distance_head.bias, -3.0)
        nn.init.constant_(self.upper_distance_head.bias, -3.0)

    def forward(
        self,
        sequence: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("LSTM input must have shape [batch, time, features]")
        if time_mask.shape != sequence.shape[:2]:
            raise ValueError("LSTM time mask must match input batch/time dimensions")
        lengths = time_mask.to(dtype=torch.int64).sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("LSTM input contains an empty sequence")
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        encoded = self.dropout(hidden[-1])
        direction_logit = self.direction_head(encoded).squeeze(-1)
        expected_return = self.return_head(encoded).squeeze(-1)
        lower_distance = F.softplus(
            self.lower_distance_head(encoded).squeeze(-1)
        )
        upper_distance = F.softplus(
            self.upper_distance_head(encoded).squeeze(-1)
        )
        lower = expected_return - lower_distance
        upper = expected_return + upper_distance
        return {
            "direction_logit": direction_logit,
            "bullish_probability": torch.sigmoid(direction_logit),
            "expected_excess_return": expected_return,
            "return_interval_lower": lower,
            "return_interval_upper": upper,
            "risk_scale": (upper - lower) / 2.0,
        }


def _pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def multi_head_lstm_loss(
    outputs: dict[str, torch.Tensor],
    *,
    direction_target: torch.Tensor,
    excess_return_target: torch.Tensor,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    huber_delta: float = 0.02,
    direction_weight: float = 1.0,
    return_weight: float = 1.0,
    interval_weight: float = 0.5,
) -> LstmLoss:
    direction = F.binary_cross_entropy_with_logits(
        outputs["direction_logit"], direction_target
    )
    expected_return = F.huber_loss(
        outputs["expected_excess_return"],
        excess_return_target,
        delta=huber_delta,
    )
    interval = _pinball_loss(
        outputs["return_interval_lower"],
        excess_return_target,
        lower_quantile,
    ) + _pinball_loss(
        outputs["return_interval_upper"],
        excess_return_target,
        upper_quantile,
    )
    total = (
        direction_weight * direction
        + return_weight * expected_return
        + interval_weight * interval
    )
    return LstmLoss(
        total=total,
        direction=direction,
        expected_return=expected_return,
        interval=interval,
    )

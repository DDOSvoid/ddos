from __future__ import annotations

import torch

from src.prediction.timeseries.lstm_model import (
    SmallLstmMultiHead,
    multi_head_lstm_loss,
)


def test_small_lstm_outputs_registered_shapes_and_ordered_interval():
    torch.manual_seed(20260824)
    model = SmallLstmMultiHead(
        input_size=10,
        hidden_size=16,
        external_dropout=0.10,
    )
    sequence = torch.randn(4, 20, 10)
    mask = torch.ones(4, 20, dtype=torch.bool)
    outputs = model(sequence, mask)

    assert model.encoder.num_layers == 1
    assert not model.encoder.bidirectional
    assert outputs["bullish_probability"].shape == (4,)
    assert torch.all(outputs["bullish_probability"] >= 0)
    assert torch.all(outputs["bullish_probability"] <= 1)
    assert torch.all(
        outputs["return_interval_lower"]
        <= outputs["expected_excess_return"]
    )
    assert torch.all(
        outputs["expected_excess_return"]
        <= outputs["return_interval_upper"]
    )
    assert torch.all(outputs["risk_scale"] >= 0)


def test_multi_head_loss_backpropagates_through_all_heads():
    torch.manual_seed(20260824)
    model = SmallLstmMultiHead(
        input_size=10,
        hidden_size=16,
        external_dropout=0.0,
    )
    sequence = torch.randn(8, 20, 10)
    mask = torch.ones(8, 20, dtype=torch.bool)
    target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    returns = torch.linspace(-0.04, 0.04, 8)
    outputs = model(sequence, mask)
    losses = multi_head_lstm_loss(
        outputs,
        direction_target=target,
        excess_return_target=returns,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert model.direction_head.weight.grad is not None
    assert model.return_head.weight.grad is not None
    assert model.lower_distance_head.weight.grad is not None
    assert model.upper_distance_head.weight.grad is not None

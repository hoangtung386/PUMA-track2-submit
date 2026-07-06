import torch

from prometheus.config import PrometheusModelConfig
from prometheus.models import MultitaskOutput, PrometheusNet


def _config(context_enabled: bool = True) -> PrometheusModelConfig:
    return PrometheusModelConfig(
        encoder_dims=[16, 32, 64, 128],
        encoder_depths=[1, 1, 1, 1],
        tissue_decoder_depths=[1, 1, 1],
        context_enabled=context_enabled,
    )


def test_prometheus_net_typed_output_and_shapes() -> None:
    model = PrometheusNet(_config())
    output = model(torch.randn(2, 3, 65, 79))
    assert isinstance(output, MultitaskOutput)
    assert output.tissue_logits.shape == (2, 6, 65, 79)
    assert output.nuclei_center_logits.shape == (2, 10, 24, 24)
    assert output.nuclei_offsets.shape == (2, 2, 24, 24)
    assert "context_gate_mean" in output.auxiliary


def test_context_can_be_removed_without_attention_fallback() -> None:
    model = PrometheusNet(_config(context_enabled=False))
    assert model.context_fusion is None
    assert not any("attention" in name or "moe" in name for name, _ in model.named_modules())
    output = model(torch.randn(1, 3, 64, 64))
    assert output.auxiliary == {}


def test_nuclei_head_accepts_non_power_of_two_channels() -> None:
    config = PrometheusModelConfig(
        encoder_dims=[40, 80, 160, 320],
        encoder_depths=[1, 1, 1, 1],
        tissue_decoder_depths=[1, 1, 1],
    )
    output = PrometheusNet(config)(torch.randn(1, 3, 64, 64))
    assert output.nuclei_center_logits.shape == (1, 10, 16, 16)

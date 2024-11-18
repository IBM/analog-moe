import torch
from analog_moe import AnalogSigmaMoELayerAIHWKITLightning
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig


def test_load_into_traceable():
    d_model = 10

    kwargs = {
        "d_model": d_model,
        "n_experts": 5,
        "expert_size": 100,
        "k": 2,
        "dropout": 0,
        "bias": True,
    }

    rpu_config = TorchInferenceRPUConfig()

    layer = AnalogSigmaMoELayerAIHWKITLightning(rpu_config=rpu_config, **kwargs)
    layer_traceable = AnalogSigmaMoELayerAIHWKITLightning(
        rpu_config=rpu_config, traceable=True, **kwargs
    )

    layer.eval()
    layer_traceable.eval()

    layer_traceable.load_state_dict(layer.state_dict())

    layer.load_state_dict(layer_traceable.state_dict())

    input = torch.randn(1, 10, d_model)
    out_traceable = layer_traceable(input)
    out = layer(input)

    assert torch.allclose(out[0], out_traceable[0], atol=1e-5)


if __name__ == "__main__":
    test_load_into_traceable()

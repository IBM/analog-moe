import os
from typing import Tuple
from pytest import mark, fixture
from unittest import SkipTest

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import (
    allclose,
    randn,
    zeros,
    float32,
    Tensor,
    manual_seed,
    sum,
)
from torch.nn import Linear
import torch.distributed
import torch.nn.functional as F

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightClipType, WeightModifierType
from aihwkit_lightning.nn.conversion import convert_to_analog
from sigma_moe.moe_layer import SigmaMoELayer as HFSigmaMoELayer

from analog_moe import AnalogSigmaMoELayerAIHWKITLightning


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


class SigmaMoELayer(torch.nn.Module):
    """
    Naive implementation of MoE layer using torch Linear layers.
    """

    def __init__(self, d_model: int, n_experts: int, expert_size: int, k: int):
        super().__init__()
        self.k_dim = d_model
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.n_heads = k

        self.keys = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=self.k_vec_dim,
                    out_features=self.expert_size,
                    bias=False,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.values = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=self.expert_size,
                    out_features=self.k_vec_dim,
                    bias=False,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.expert_sel = torch.nn.Linear(
            in_features=self.k_vec_dim, out_features=self.n_experts, bias=False
        )

    def compute_scores(self, input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = input.shape
        scores = torch.zeros((bsz, seq_len, self.expert_size)).to(input.device)
        for b in range(bsz):
            for s in range(seq_len):
                token = input[b, s]
                expert_idx = index[b, s]
                scores[b, s] = self.keys[expert_idx](token)
        scores = F.relu(scores)
        return scores

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        into_router = input.clone()
        sel = self.expert_sel(into_router)
        sel = torch.sigmoid(sel)
        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        # "Up-projection" layer for each head
        input_into_up_proj = input.clone()
        scores_l = [
            self.compute_scores(input_into_up_proj, sel_index[..., h].long())
            for h in range(sel_index.shape[-1])
        ]

        # Down projection layer for each head
        res = torch.zeros_like(input)
        bsz, seq_len, _ = input.shape
        for h, scores in enumerate(scores_l):
            sel_index_h = sel_index[..., h]
            sel_val_h = sel_val[..., h]
            for b in range(bsz):
                for s in range(seq_len):
                    expert_idx = sel_index_h[b, s]
                    res[b, s] = res[b, s] + sel_val_h[b, s] * self.values[expert_idx](
                        scores[b, s]
                    )

        # # for debugging: return the middle scores (after ReLU)
        # scores_torch = torch.empty((*res.shape[:2], len(scores_l), scores_l[0].shape[-1]), device=res.device)
        # for h, scores in enumerate(scores_l):
        #     scores_torch[..., h, :] = scores

        return res, 0.0

    def set_from_hf(self, hf_moe) -> None:
        self.expert_sel.weight.data = hf_moe.expert_sel.weight.data
        for e in range(self.n_experts):
            self.keys[e].weight.data = hf_moe.keys[e].T
            self.values[e].weight.data = hf_moe.values[e].T


@fixture(scope="module", name="weight_modifier_type")
def fixture_weight_modifier_type(request) -> WeightModifierType:
    """Weight Modifier type"""
    return request.param

@fixture(scope="module", name="weight_clip_type")
def fixture_weight_clip_type(request) -> WeightClipType:
    """Weight Clip type"""
    return request.param

@fixture(scope="module", name="max_inp_size")
def fixture_max_inp_size(request) -> int:
    """Maximum input size parameter"""
    return request.param


@fixture(scope="module", name="ir_enable_inp_res")
def fixture_ir_enable_inp_res(request) -> Tuple[bool, float]:
    """Combination of ir_enable and inp_res parameters"""
    return request.param


@fixture(scope="module", name="ir_init_value")
def fixture_ir_init_value(request) -> float:
    """IR initialization value parameter"""
    return request.param


@fixture(scope="module", name="ir_init_std_alpha")
def fixture_ir_init_std_alpha(request) -> float:
    """IR initialization alpha parameter"""
    return request.param


@fixture(scope="module", name="adc_config")
def fixture_adc_config(request) -> Tuple[float, float]:
    """Tuple of out_bound, out_res for ADC"""
    return request.param


@fixture(scope="module", name="rpu_config")
def fixture_rpus(
    max_inp_size: int,
    ir_enable_inp_res: Tuple[bool, float],
    ir_init_value: float,
    ir_init_std_alpha: float,
    adc_config: Tuple[float, float],
) -> TorchInferenceRPUConfig:
    """Fixture for initializing rpus globally for all tests that need them"""
    ir_enable, inp_res = ir_enable_inp_res
    out_bound, out_res = adc_config
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.inp_res = inp_res
    rpu_config.forward.out_res = out_res
    rpu_config.forward.out_bound = out_bound
    rpu_config.forward.out_noise = 0.0
    rpu_config.mapping.max_input_size = max_inp_size
    rpu_config.pre_post.input_range.enable = ir_enable
    rpu_config.pre_post.input_range.learn_input_range = True
    rpu_config.pre_post.input_range.init_value = ir_init_value
    # this works. however, the groundtruth sends one vector at a time,
    # thus updating it each time compared to our case where we just
    # update once per batch which is correct
    rpu_config.pre_post.input_range.init_from_data = -1
    rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
    return rpu_config


@mark.parametrize("max_inp_size", [-1], indirect=True)
@mark.parametrize("ir_enable_inp_res", [(False, -1), (True, 2**8 - 2), (True, 1 / (2**8 - 2))], ids=str, indirect=True)
@mark.parametrize("ir_init_value", [2.0], indirect=True)
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0], indirect=True)
@mark.parametrize("adc_config", [(-1, -1)], ids=str, indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_analog_vs_normal_gradient(
    device: torch_device,
    dtype: torch_dtype,
    rpu_config: TorchInferenceRPUConfig,
):
    """
    Test input output correctness of MoE layer.
    Also check correctness of gradients w.r.t. inputs, weights.
    """

    if device == "cpu":
        raise SkipTest("CPU currently not supported")

    manual_seed(0)
    d_model = 128
    n_experts = 4
    expert_size = 64
    seq_len = 10
    bsz = 10
    k = 2
    hf_moe = HFSigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).to(device=device, dtype=dtype)
    moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).to(device=device, dtype=dtype)
    moe.set_from_hf(hf_moe)

    fill_data = randn(bsz, seq_len, d_model).to(device=device, dtype=dtype)

    inp = zeros(bsz, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    inp.data = fill_data

    fast_inp = zeros(bsz, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    fast_inp.data = fill_data.data

    # now, we convert the nn.Linear version to an analog one
    analog_moe = convert_to_analog(moe, rpu_config=rpu_config)

    # make contiguous for triton mode
    for analog_layer in analog_moe.analog_layers():
        analog_layer.weight.data = analog_layer.weight.contiguous()

    fast_analog_moe = convert_to_analog(
        hf_moe,
        rpu_config=rpu_config,
        conversion_map={
            Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKITLightning,
        },
    )

    os.environ["_AIHWKIT_NO_ROUNDING"] = "1"
    os.environ["AIHWKIT_TESTING"] = "1"

    analog_x: Tensor
    analog_x, _ = analog_moe(inp)
    fast_analog_x: Tensor
    fast_analog_x, _ = fast_analog_moe(fast_inp)
    assert allclose(analog_x, fast_analog_x, atol=1e-4)

    loss = sum(analog_x)
    loss.backward()

    fast_loss = sum(fast_analog_x)
    fast_loss.backward()

    # test grad of the values
    grad_values_exp1 = analog_moe.values[0].weight.grad
    assert allclose(grad_values_exp1.T, fast_analog_moe.values.grad[0], atol=1e-4)

    # test grad of the keys, if passing means that dL / dp1 (before ReLU) is correct
    grad_values_exp1 = analog_moe.keys[0].weight.grad
    assert allclose(grad_values_exp1.T, fast_analog_moe.keys.grad[0], atol=1e-4)

    # test grad of the inputs
    # abs-max introduces some discrepancy, which is why we are using 1e-3 for this
    assert allclose(inp.grad, fast_inp.grad, atol=1e-3)

    del os.environ["_AIHWKIT_NO_ROUNDING"]
    del os.environ["AIHWKIT_TESTING"]


@mark.parametrize(
    "weight_clip_type",
    [WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL, WeightClipType.NONE],
    ids=str,
    indirect=True
)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_clipping(
        weight_clip_type: WeightClipType,
        device: torch_device,
        dtype: torch_dtype
):
    manual_seed(0)
    d_model = 128
    n_experts = 4
    expert_size = 64
    k = 2
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.clip.type = weight_clip_type
    rpu_config.clip.sigma = 2.0
    moe = HFSigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).to(device=device, dtype=dtype)
    analog_moe = convert_to_analog(
        moe,
        rpu_config=rpu_config,
        conversion_map={
            Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKITLightning
        }
    )
    for analog_layer in analog_moe.analog_layers():
        analog_layer.clip_weights()


@mark.parametrize(
    "weight_modifier_type",
    [
        WeightModifierType.ADD_NORMAL, WeightModifierType.ADD_NORMAL_PER_CHANNEL,
        WeightModifierType.DISCRETIZE, WeightModifierType.DISCRETIZE_ADD_NORMAL,
        WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL, WeightModifierType.DISCRETIZE_PER_CHANNEL
    ],
    ids=str,
    indirect=True
)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_weight_modifier(
        weight_modifier_type: WeightModifierType,
        device: torch_device,
        dtype: torch_dtype
):
    manual_seed(0)
    d_model = 128
    n_experts = 4
    expert_size = 64
    k = 2
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.modifier.type = weight_modifier_type
    rpu_config.modifier.res = 2**8 - 2
    rpu_config.modifier.std_dev = 2.0
    moe = HFSigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).to(device=device, dtype=dtype)
    analog_moe = convert_to_analog(
        moe,
        rpu_config=rpu_config,
        conversion_map={
            Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKITLightning
        }
    )
    inp_data = randn(5, 10, d_model).to(device=device, dtype=dtype)
    analog_moe(inp_data)
    


if __name__ == "__main__":

    test_clipping(WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL, "cuda", float32)

    # rpu_config = fixture_rpus(
    #     max_inp_size=-1,
    #     ir_enable_inp_res=(False, -1),
    #     ir_init_value=2.0,
    #     ir_init_std_alpha=3.0,
    #     adc_config=(-1, -1),
    # )

    # test_analog_vs_normal_gradient(
    #     device=torch_device("cuda"),
    #     dtype=float32,
    #     rpu_config=rpu_config
    # )

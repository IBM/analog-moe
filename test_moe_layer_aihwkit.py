from pytest import mark
from typing import Tuple
import torch
import torch.distributed
import torch.nn.functional as F

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.configs import (
    BoundManagementType,
    NoiseManagementType,
    WeightClipType,
    WeightRemapType,
)
from aihwkit.optim import AnalogSGD
from aihwkit.nn.conversion import convert_to_analog
from sigma_moe.moe_layer import SigmaMoELayer as HFSigmaMoELayer

from analog_moe import AnalogSigmaMoELayerAIHWKIT


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

        # # for debugging: return the middle scores
        # scores_torch = torch.empty((*res.shape[:2], len(scores_l), scores_l[0].shape[-1]), device=res.device)
        # for h, scores in enumerate(scores_l):
        #     scores_torch[..., h, :] = scores

        return res, 0.0

    def set_from_hf(self, hf_moe) -> None:
        self.expert_sel.weight.data = hf_moe.expert_sel.weight.data
        for e in range(self.n_experts):
            self.keys[e].weight.data = hf_moe.keys[e].T
            self.values[e].weight.data = hf_moe.values[e].T


@mark.parametrize("weight_remap_columnwise", [True, False])
def test_out_noise(weight_remap_columnwise: bool):
    """
    Inject out noise and check the error
    vector's std (i.e. take the difference between
    the target and the output and compute the std).
    """
    # the rpu config we will test for
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.remap.type = (
        WeightRemapType.CHANNELWISE_SYMMETRIC
        if weight_remap_columnwise
        else WeightRemapType.LAYERWISE_SYMMETRIC
    )
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.inp_bound = -1
    rpu_config.forward.out_noise = 0.01
    rpu_config.forward.is_perfect = False
    rpu_config.pre_post.input_range.enable = False
    rpu_config.pre_post.input_range.decay = 0.0
    rpu_config.pre_post.input_range.init_from_data = 0
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0

    torch.manual_seed(0)
    d_model = 512
    n_experts = 4
    expert_size = 512
    seq_len = 100
    bsz = 100
    k = 2
    hf_moe = HFSigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()
    moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()
    hf_moe.keys.data /= hf_moe.keys.abs().amax(1).unsqueeze(1)
    hf_moe.values.data /= hf_moe.values.abs().amax(1).unsqueeze(1)
    moe.set_from_hf(hf_moe)
    analog_moe = convert_to_analog(moe, rpu_config=rpu_config)
    fast_analog_moe = convert_to_analog(
        hf_moe,
        rpu_config=rpu_config,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )
    inp = torch.randn(bsz, seq_len, d_model, device="cuda")
    fast_analog_x, _ = fast_analog_moe(inp)
    hf_x, _ = hf_moe(inp)
    analog_x, _ = analog_moe(inp)
    fast_diff = (hf_x - fast_analog_x).abs()
    diff = (hf_x - analog_x).abs()
    assert torch.allclose(diff.std(-1).mean(), fast_diff.std(-1).mean(), atol=1e-2)


@mark.parametrize("use_abs_max", [True, False])
def test_analog_vs_normal_IR_gradient(use_abs_max: bool):
    """
    Test input output correctness of MoE layer under input ranges.
    Also check correctness of gradients w.r.t. inputs, weights and
    input ranges.
    """

    # the rpu config we will test for
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = (
        NoiseManagementType.ABS_MAX if use_abs_max else NoiseManagementType.NONE
    )
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.is_perfect = False
    rpu_config.pre_post.input_range.enable = False if use_abs_max else True
    rpu_config.pre_post.input_range.decay = 0.0
    rpu_config.pre_post.input_range.init_from_data = 0
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0

    torch.manual_seed(0)
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
    ).cuda()
    moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()
    moe.set_from_hf(hf_moe)

    fill_data = torch.randn(bsz, seq_len, d_model).cuda()

    inp = torch.zeros(bsz, seq_len, d_model, requires_grad=True, device="cuda")
    inp.data = fill_data

    fast_inp = torch.zeros(bsz, seq_len, d_model, requires_grad=True, device="cuda")
    fast_inp.data = fill_data.data

    # now, we convert the torch.nn.Linear version to an analog one
    analog_moe = convert_to_analog(moe, rpu_config=rpu_config)
    # analog_moe = analog_moe.eval()

    fast_analog_moe = convert_to_analog(
        hf_moe,
        rpu_config=rpu_config,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )

    if not use_abs_max:
        # change the input ranges
        for i in range(n_experts):
            analog_moe.keys[i].analog_module.input_range.data = torch.tensor(
                [i + 1.0]
            ).cuda()
            analog_moe.values[i].analog_module.input_range.data = torch.tensor(
                [i + 1.0]
            ).cuda()

        fast_analog_moe.input_range.data[0] = torch.tensor(
            [1.0 + i for i in range(n_experts)]
        ).cuda()
        fast_analog_moe.input_range.data[1] = torch.tensor(
            [1.0 + i for i in range(n_experts)]
        ).cuda()

    analog_x, _ = analog_moe(inp)
    fast_analog_x, _ = fast_analog_moe(fast_inp)
    assert torch.allclose(analog_x, fast_analog_x, atol=1e-4)

    loss = torch.sum(analog_x)
    loss.backward()

    fast_loss = torch.sum(fast_analog_x)
    fast_loss.backward()

    # test grad of the values
    grad_values_exp1 = analog_moe.values[0].analog_module.tile.weight.grad
    assert torch.allclose(grad_values_exp1.T, fast_analog_moe.values.grad[0], atol=1e-4)

    # test grad of the keys, if passing means that dL / dp1 (before ReLU) is correct
    grad_values_exp1 = analog_moe.keys[0].analog_module.tile.weight.grad
    assert torch.allclose(grad_values_exp1.T, fast_analog_moe.keys.grad[0], atol=1e-4)

    # test grad of the inputs
    # abs-max introduces some discrepancy, which is why we are using 1e-3 for this
    assert torch.allclose(inp.grad, fast_inp.grad, atol=1e-3)

    if not use_abs_max:
        # test grad of input ranges
        keys_ir_grads = torch.tensor(
            [
                analog_moe.keys[i].analog_module.input_range.grad
                for i in range(n_experts)
            ]
        ).cuda()
        values_ir_grads = torch.tensor(
            [
                analog_moe.values[i].analog_module.input_range.grad
                for i in range(n_experts)
            ]
        ).cuda()
        assert torch.allclose(
            fast_analog_moe.input_range.grad[0], keys_ir_grads, atol=1e-4
        )
        assert torch.allclose(
            fast_analog_moe.input_range.grad[1], values_ir_grads, atol=1e-4
        )


@mark.parametrize("do_clip", [True, False])
def test_analog_optim(do_clip: bool):
    """
    Test analog optimizers.
    """
    # the rpu config we will test for
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.clip.type = (
        WeightClipType.LAYER_GAUSSIAN if do_clip else WeightClipType.NONE
    )
    rpu_config.clip.sigma = 2.5
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.inp_bound = -1
    rpu_config.forward.is_perfect = True
    rpu_config.pre_post.input_range.enable = False
    rpu_config.pre_post.input_range.decay = 0.0
    rpu_config.pre_post.input_range.init_from_data = 0
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0

    torch.manual_seed(0)
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
    ).cuda()
    moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()
    moe.set_from_hf(hf_moe)
    analog_moe = convert_to_analog(moe, rpu_config=rpu_config)
    fast_analog_moe = convert_to_analog(
        hf_moe,
        rpu_config=rpu_config,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )
    inp = torch.randn(bsz, seq_len, d_model, device="cuda")
    optim = AnalogSGD(analog_moe.parameters(), lr=1e-3)
    out, _ = analog_moe(inp)
    loss = out.mean()
    loss.backward()
    optim.step()

    fast_optim = AnalogSGD(fast_analog_moe.parameters(), lr=1e-3)
    fast_out, _ = fast_analog_moe(inp)
    fast_loss = fast_out.mean()
    fast_loss.backward()
    fast_optim.step()

    # are the grads the same?
    assert torch.allclose(
        analog_moe.keys[0].analog_module.tile.weight.grad.T,
        fast_analog_moe.keys.grad[0],
        atol=1e-4,
    )
    assert torch.allclose(
        analog_moe.values[0].analog_module.tile.weight.grad.T,
        fast_analog_moe.values.grad[0],
        atol=1e-4,
    )
    # was the update correct?
    assert torch.allclose(
        analog_moe.keys[0].analog_module.tile.weight.T,
        fast_analog_moe.keys[0],
        atol=1e-4,
    )
    assert torch.allclose(
        analog_moe.values[0].analog_module.tile.weight.T,
        fast_analog_moe.values[0],
        atol=1e-4,
    )


def test_load_and_state_dict():
    """
    Test loading and saving of analog moe layers.
    """
    # the rpu config we will test for
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.inp_bound = -1
    rpu_config.forward.is_perfect = True
    rpu_config.pre_post.input_range.enable = False
    rpu_config.pre_post.input_range.decay = 0.0
    rpu_config.pre_post.input_range.init_from_data = 0
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0

    torch.manual_seed(0)
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
    ).cuda()
    moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()

    moe.set_from_hf(hf_moe)
    analog_moe = convert_to_analog(moe, rpu_config=rpu_config)
    fast_analog_moe = convert_to_analog(
        hf_moe,
        rpu_config=rpu_config,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )
    inp = torch.randn(bsz, seq_len, d_model, device="cuda")
    out, _ = analog_moe(inp)
    fast_out, _ = fast_analog_moe(inp)
    assert torch.allclose(out, fast_out, atol=1e-4)
    analog_moe_sd = analog_moe.state_dict()
    analog_moe.load_state_dict(analog_moe_sd)
    fast_analog_moe_sd = fast_analog_moe.state_dict()
    fast_analog_moe.load_state_dict(fast_analog_moe_sd)
    out, _ = analog_moe(inp)
    fast_out, _ = fast_analog_moe(inp)
    assert torch.allclose(out, fast_out, atol=1e-4)


def test_to_cuda():
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.inp_bound = -1
    rpu_config.forward.is_perfect = True
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0

    class TestModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.linear = torch.nn.Linear(10, 10)
            self.moe1 = SigmaMoELayer(d_model=10, n_experts=4, expert_size=10, k=2)
            self.moe2 = SigmaMoELayer(d_model=10, n_experts=4, expert_size=10, k=2)

        def forward(self, x):
            return self.moe1(self.linear(x))

    model = TestModel()
    model = convert_to_analog(
        model,
        rpu_config=rpu_config,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            HFSigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )
    model = model.to(torch.device("cuda"))


if __name__ == "__main__":
    test_analog_vs_normal_IR_gradient(True)
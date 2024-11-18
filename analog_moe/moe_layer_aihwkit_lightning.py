from typing import Optional, OrderedDict, Union
import re
from functools import partial
from torch import Tensor, no_grad, randn_like, randn, full, tensor, zeros, stack, concat
from torch.nn import Linear, Parameter
import torch.nn.functional as F

from sigma_moe.moe_layer import SigmaMoELayer

from aihwkit_lightning.nn.modules.base import AnalogLayerBase
from aihwkit_lightning.simulator.configs.configs import TorchInferenceRPUConfig
from aihwkit_lightning.exceptions import ConfigError
from aihwkit_lightning.simulator.parameters.enums import (
    WeightModifierType,
    WeightClipType,
)
from aihwkit_lightning.simulator.parameters import WeightModifierParameter
from aihwkit_lightning.nn.modules.torch_utils.torch_linear import UniformQuantize
from aihwkit_lightning.nn.conversion import convert_to_analog

HAS_TRITON = True
try:
    from .triton_src.cvmm import CVMM, CVMMSel, cvmm_std
except:
    print("WARNING: Could not load triton.")
    HAS_TRITON = False


class MoEConifgError(Exception):
    """Exceptions related to MoE configuration."""


def non_traceable_to_traceable(state_dict: OrderedDict, prefix: str):
    """Convert non-traceable state dict into traceable state dict in-place"""
    input_range = state_dict[prefix + "input_range"]
    update_idx = state_dict[prefix + "input_range_update_idx"]
    n_experts = input_range.size(1)
    device = input_range.device
    dtype = input_range.dtype
    for i in range(n_experts):
        state_dict[prefix + f"keys.{i}.input_range"] = input_range[0, i].view(
            1,
        )
        state_dict[prefix + f"keys.{i}.input_range_update_idx"] = update_idx.view(
            1,
        )
        state_dict[prefix + f"keys.{i}.x_min"] = zeros((1,), device=device, dtype=dtype)
        state_dict[prefix + f"keys.{i}.x_max"] = zeros((1,), device=device, dtype=dtype)
        state_dict[prefix + f"values.{i}.input_range"] = input_range[1, i].view(
            1,
        )
        state_dict[prefix + f"values.{i}.input_range_update_idx"] = update_idx.view(
            1,
        )
        state_dict[prefix + f"values.{i}.x_min"] = zeros(
            (1,), device=device, dtype=dtype
        )
        state_dict[prefix + f"values.{i}.x_max"] = zeros(
            (1,), device=device, dtype=dtype
        )

    state_dict.pop(prefix + "input_range")
    state_dict.pop(prefix + "input_range_update_idx")


def traceable_to_non_traceable(state_dict: OrderedDict, prefix: str):
    """Convert traceable to non-traceable state dict"""
    key_value_x_names = [
        k
        for k in state_dict
        if re.match(rf"^{re.escape(prefix)}(keys|values)\.\d+\.(x_min|x_max)$", k)
    ]
    update_idx_names = [
        k
        for k in state_dict
        if re.match(
            rf"^{re.escape(prefix)}(keys|values)\.\d+\.input_range_update_idx$", k
        )
    ]
    key_ir_names = [
        k
        for k in state_dict
        if re.match(rf"^{re.escape(prefix)}keys\.\d+\.input_range$", k)
    ]
    values_ir_names = [
        k
        for k in state_dict
        if re.match(rf"^{re.escape(prefix)}values\.\d+\.input_range$", k)
    ]
    input_range = stack(
        [
            concat([state_dict[k] for k in key_ir_names]),
            concat([state_dict[k] for k in values_ir_names]),
        ]
    )
    keys_to_delete = [
        *key_ir_names,
        *values_ir_names,
        *key_value_x_names,
        *update_idx_names,
    ]
    state_dict[prefix + "input_range"] = input_range
    state_dict[prefix + "input_range_update_idx"] = state_dict[update_idx_names[0]]
    for key_to_delete in keys_to_delete:
        state_dict.pop(key_to_delete)


def load_state_dict_pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    traceable,
):
    has_ir = state_dict._metadata[prefix]["rpu_config"].pre_post.input_range.enable
    if not has_ir:
        return

    if has_ir and prefix + "input_range" in state_dict:
        state_dict_is_for_non_traceable = True
    else:
        state_dict_is_for_non_traceable = False

    if traceable and state_dict_is_for_non_traceable:
        # convert state dict to traceable
        non_traceable_to_traceable(state_dict=state_dict, prefix=prefix)
    elif not traceable and not state_dict_is_for_non_traceable:
        traceable_to_non_traceable(state_dict=state_dict, prefix=prefix)


class AnalogSigmaMoELayerAIHWKITLightning(AnalogLayerBase, SigmaMoELayer):
    def __init__(
        self, rpu_config: Optional[TorchInferenceRPUConfig] = None, *args, **kwargs
    ):
        SigmaMoELayer.__init__(self, *args, **kwargs)
        AnalogLayerBase.__init__(self)

        if rpu_config is None:
            self.rpu_config = TorchInferenceRPUConfig()
        else:
            assert isinstance(
                rpu_config, TorchInferenceRPUConfig
            ), "rpu_config must be a TorchInferenceRPUConfig"
            self.rpu_config = rpu_config

        # TODO do some checks here for the things we support
        # TODO also add class function that raise MoEConifgError that
        # lists all the things the user did wrong with the rpu config
        assert (
            self.rpu_config.forward.out_bound <= 0
        ), "out_bound must be <= 0, i.e. unbounded"
        assert (
            self.rpu_config.forward.out_res == -1
        ), "out_res must be -1.0, i.e. full precision"
        assert (
            self.rpu_config.mapping.max_input_size <= 0
        ), "max_input_size must be <= 0"

        self._register_load_state_dict_pre_hook(
            partial(load_state_dict_pre_hook, traceable=self.traceable)
        )
        if self.traceable:
            convert_to_analog(
                self, rpu_config=rpu_config, inplace=True, ensure_analog_root=False
            )
        else:
            # initialize the input ranges
            self.input_range = None
            ir_params = self.rpu_config.pre_post.input_range
            if ir_params.enable:
                self.input_range_update_idx = Parameter(
                    tensor(0.0, requires_grad=False)
                )
                if ir_params.learn_input_range:
                    self.input_range = Parameter(
                        full(
                            (2, self.n_experts),
                            ir_params.init_value,
                            requires_grad=True,
                        )
                    )
                else:
                    input_range = full(
                        (2, self.n_experts),
                        ir_params.init_value,
                        requires_grad=False,
                    )
                    if hasattr(self, "input_range") and self.input_range is None:
                        delattr(self, "input_range")
                    self.register_buffer("input_range", input_range)  # type: ignore

            self.expert_sel = convert_to_analog(
                self.expert_sel,
                rpu_config=rpu_config,
                inplace=False,
                ensure_analog_root=False,
            )
            self.set_weights(
                expert_sel=self.expert_sel,
                keys=self.keys,
                values=self.values,
                bias=self.bias,
                o_bias=self.o_bias,
            )

    @classmethod
    def to_digital(cls, module: "AnalogSigmaMoELayerAIHWKITLightning") -> SigmaMoELayer:
        """Return a SigmaMoELayer layer from an AnalogSigmaMoELayerAIHWKITLightning layer.

        Args:
            module: The analog module to convert.

        Returns:
            a SigmaMoELayer with the same dimension and weights
            as the analog version.
        """
        digital_layer = SigmaMoELayer(
            d_model=module.k_dim,
            n_experts=module.n_experts,
            expert_size=module.expert_size,
            k=module.n_heads,
            dropout=module.dropout,
            selection_mode=module.selection_mode,
            activation_after_topk=module.activation_after_topk,
            activation=module.activation,
            bias=module.bias is not None,
            v_dim=module.v_dim,
            sinkhorn_n_iters=module.sinkhorn_n_iters,
            expert_dropout=module.expert_dropout,
        )
        digital_layer.keys.data = module.keys.data.detach().clone()
        digital_layer.values.data = module.values.data.detach().clone()
        digital_layer.expert_sel.load_state_dict(module.expert_sel.state_dict())
        if module.bias is not None:
            digital_layer.bias.data = module.bias.data.detach().clone()
            digital_layer.o_bias.data = module.o_bias.data.detach().clone()
        return digital_layer.to(device=module.keys.device, dtype=module.values.dtype)

    @classmethod
    def move_to_meta(cls, module: "AnalogSigmaMoELayerAIHWKITLightning"):
        """Move the module to the meta class.

        This is used to move the module to the meta class. This is
        useful for the conversion of the module to analog.

        Args:
            module: The module to move to the meta class.

        """
        module = module.to(device="meta")

    @classmethod
    def from_digital(
        cls,
        module: SigmaMoELayer,
        rpu_config: TorchInferenceRPUConfig,
    ) -> "AnalogSigmaMoELayerAIHWKITLightning":
        analog_layer = cls(
            rpu_config=rpu_config,
            d_model=module.k_dim,
            n_experts=module.n_experts,
            expert_size=module.expert_size,
            k=module.n_heads,
            dropout=module.dropout,
            selection_mode=module.selection_mode,
            activation_after_topk=module.activation_after_topk,
            activation=module.activation,
            bias=module.bias,
            v_dim=module.v_dim,
            sinkhorn_n_iters=module.sinkhorn_n_iters,
            expert_dropout=module.expert_dropout,
        )
        analog_layer.set_weights(
            expert_sel=module.expert_sel,
            keys=module.keys,
            values=module.values,
            bias=module.bias,
            o_bias=module.o_bias,
        )
        return analog_layer.to(module.keys.device)

    def set_weights(
        self,
        expert_sel: Linear,
        keys: Tensor,
        values: Tensor,
        bias: Tensor | None = None,
        o_bias: Tensor | None = None,
    ) -> None:
        """Sets the weights and bias tensors. Creates a copy of the tensors.

        Args:
            expert_sel: the weight tensor
        """
        self.expert_sel.load_state_dict(expert_sel.state_dict())
        self.keys.data = keys.detach().clone()
        self.values.data = values.detach().clone()
        if bias is not None:
            self.bias.data = bias.detach().clone()
        if o_bias is not None:
            self.o_bias.data = o_bias.detach().clone()

    @staticmethod
    def modify_weight(
        inp_weight: Tensor,
        modifier: WeightModifierParameter,
    ) -> Tensor:
        """Modifies weights in-place, so .clone() before passing the weights here.

        Args:
            inp_weight: Input weights.
            assumed_wmax: Assumed maximum weight value.
            modifier: WeightModifierParameter.

        Raises:
            ConfigError: Unsupported/unknown weight modifier type.

        Returns:
            Weights with noise injected.
        """
        if modifier.type == WeightModifierType.NONE:
            return inp_weight

        if modifier.type in [
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ]:
            # need per column wmax
            assumed_wmax = inp_weight.abs().amax(1)
            assumed_wmax = assumed_wmax.unsqueeze(1)  # [n_experts, 1, n_columns]
        else:
            # [n_experts, 1, 1]
            assumed_wmax = (
                inp_weight.view(inp_weight.size(0), -1).abs().amax(-1).view(-1, 1, 1)
            )

        if modifier.type in [
            WeightModifierType.DISCRETIZE,
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ]:
            res = modifier.res
            n_states = max(res, 1 / res)
            # assumed_wamax.item() would result in fp16 imprecision
            res = assumed_wmax / n_states  # type: ignore[assignment]

        if modifier.type in [
            WeightModifierType.DISCRETIZE,
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
        ]:
            # - Discretize the weights on the fly and backprob through them
            inp_weight = UniformQuantize.apply(inp_weight, res, 1.0, True)
        elif modifier.type in [
            WeightModifierType.ADD_NORMAL,
            WeightModifierType.ADD_NORMAL_PER_CHANNEL,
        ]:
            with no_grad():
                noise = modifier.std_dev * assumed_wmax * randn_like(inp_weight)
            inp_weight = inp_weight + noise
        elif modifier.type in [
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ]:
            inp_weight = UniformQuantize.apply(inp_weight, res, 1.0, True)
            with no_grad():
                noise = modifier.std_dev * assumed_wmax * randn_like(inp_weight)
            inp_weight = inp_weight + noise
        else:
            raise ConfigError(f"Weight modifier {modifier} not supported")
        return inp_weight

    def compute_scores(
        self,
        inp: Tensor,
        index: Union["CVMMSel", Tensor],
        expert_scores: Optional[Tensor] = None,
    ) -> Tensor:
        IS_CUDA = inp.is_cuda
        if IS_CUDA:
            scores = self.cvmm_wrapper(inp, index, self.keys)
            if self.bias is not None:
                scores = scores + self.bias[index.raw_sel]
        else:
            raise MoEConifgError(
                "Analog MoE executed on CPU in non-traceable mode. Please instantiate in "
                ""
                """traceable=True mode or run on GPU."""
            )

        scores = self.activation(scores)
        if expert_scores is not None:
            scores = scores * expert_scores[..., None]

        if self.dropout > 0:
            # Standard dropout on the "up-projected scores"
            scores = F.dropout(scores, self.dropout, training=self.training)

        return scores

    def cvmm_wrapper(self, inputs: Tensor, sel_indices: "CVMMSel", weights: Tensor):
        """
        TODO

        Args:
            inputs (Tensor): Shape [bsz, seq_len, d_model] or [bsz, seq_len, top_k, d_ff]
            sel_indices (CVMMSel): See `transformers.models.sigma_moe.triton_src.CVMMSel`
            weights (Tensor): Shape [n_experts, d_model, d_ff] or [n_experts, d_ff, d_model]

        Returns:
            _type_: _description_
        """
        broadcasted_input_ranges = None
        if self.input_range is not None:
            # scale the input according to the input range
            # when the input is just [bsz, seq_len, d_model] then we are doing the first MVM, i.e. the up-proj.
            is_up_projection = inputs.ndim == 3
            ir_idx = 0 if is_up_projection else 1

            # maybe adapt the input ranges here
            if self.training:
                ir_params = self.rpu_config.pre_post.input_range
                idx = self.input_range_update_idx
                if idx < ir_params.init_from_data:
                    stds = cvmm_std(
                        inputs, sel_indices.sel_index, sel_indices.sel, self.n_experts
                    )
                    if (stds > 0.0).any():
                        self.input_range.data[ir_idx] = (
                            self.input_range.data[ir_idx][stds > 0] * idx
                            + ir_params.init_std_alpha * stds[stds > 0]
                        ) / (idx + 1)
                        self.input_range_update_idx.data += 1
                    self.input_range.data = self.input_range.data.abs()

            input_ranges = self.input_range[ir_idx]
            broadcasted_input_ranges = input_ranges[sel_indices.sel]

        # what is the inp_res?
        inp_res = self.rpu_config.forward.inp_res
        if inp_res > 0:
            # yields 1 / 127. for inp_res = 2**8-2
            inp_res = 2.0 / inp_res if inp_res > 1.0 else 2.0 * inp_res
        else:
            inp_res = -1

        modified_weights = weights
        apply_weight_modifier = (
            self.training or self.rpu_config.modifier.enable_during_test
        ) and self.rpu_config.modifier.type != WeightModifierType.NONE
        if apply_weight_modifier:
            modified_weights = weights.clone()

        if apply_weight_modifier:
            # weight noise injection
            modified_weights = AnalogSigmaMoELayerAIHWKITLightning.modify_weight(
                modified_weights, self.rpu_config.modifier
            )

        out_noise = None
        if self.training and self.rpu_config.forward.out_noise > 0:
            # [bsz, seq_len, top-k, d_out]
            out_noise = randn(
                (*inputs.shape[:2], self.n_heads, weights.shape[-1]),
                device=inputs.device,
            )
            # the inputs into the MVM will be in [-1, 1] range, but the weights are not normalized
            # so we need to scale the noise by the abs max
            if self.rpu_config.forward.out_noise_per_channel:
                # scale by abs_max of weight channels
                assumed_wmax = weights.abs().amax(1)
            else:
                # scale by abs_max of weight layer
                assumed_wmax = (
                    weights.view(weights.size(0), -1).abs().amax(-1).unsqueeze(-1)
                )
            out_noise = (
                out_noise
                * self.rpu_config.forward.out_noise
                * assumed_wmax[sel_indices.raw_sel]
            )

        output = CVMM.apply(
            inputs,
            sel_indices.sel_index,
            sel_indices.sel,
            modified_weights,
            inp_res,
            broadcasted_input_ranges,
            sel_indices.out_index,
            sel_indices.reduction_weight,
            out_noise,
            self.rpu_config.pre_post.input_range,
            self.rpu_config.forward,
        )
        return output

    def post_update_step(self) -> None:
        """
        Clip weights after weights have been updated.
        """
        if (
            hasattr(self.rpu_config, "clip")
            and self.rpu_config.clip.type != WeightClipType.NONE
        ):
            self.clip_weights(self.rpu_config.clip)

    @no_grad()
    def clip_weights(
        self,
    ) -> None:
        """Clip the weights."""
        clip_type = self.rpu_config.clip.type
        clip_sigma = self.rpu_config.clip.sigma

        if clip_type == WeightClipType.NONE:
            return
        assert clip_sigma > 0, "Clip sigma must be greater than 0"
        sigma_std_keys = clip_sigma * self.keys.std(
            (1, 2) if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
        )
        sigma_std_values = clip_sigma * self.values.std(
            (1, 2) if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
        )
        if clip_type in [
            WeightClipType.LAYER_GAUSSIAN,
            WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
        ]:
            self.keys.data.clamp_(-sigma_std_keys, sigma_std_keys)
            self.values.data.clamp_(-sigma_std_values, sigma_std_values)
        else:
            raise ValueError(f"Unknown clip type {clip_type}")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # pylint: disable=protected-access
        destination._metadata[prefix.split(".")[0]]["rpu_config"] = self.rpu_config

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if "rpu_config" in local_metadata:
            self.rpu_config = local_metadata["rpu_config"]

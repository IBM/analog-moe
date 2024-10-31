from typing import NamedTuple, Optional, Type, Tuple
from collections import OrderedDict
import torch
from torch import Tensor, dtype
from torch import device as torch_device
from sigma_moe.moe_layer import SigmaMoELayer

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.configs.configs import TorchInferenceRPUConfig
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.parameters.enums import (
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    BoundManagementType,
)
from aihwkit.simulator.parameters import (
    WeightModifierParameter,
    WeightClipParameter,
)

from .triton_src.cvmm import CVMM, CVMMSel, cvmm_std

class MoEConifgError(Exception):
    """Exceptions related to MoE configuration."""


class AnalogSigmaMoELayerAIHWKIT(AnalogLayerBase, SigmaMoELayer):
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
        assert self.rpu_config.forward.inp_bound in [
            -1,
            1.0,
        ], "inp_bound must be 1.0 or off (=-1)"
        assert (
            self.rpu_config.forward.out_bound == -1
        ), "out_bound must be -1.0, i.e. unbounded"
        assert (
            self.rpu_config.forward.out_res == -1
        ), "out_res must be -1.0, i.e. full precision"
        assert self.rpu_config.mapping.max_input_size == 0, "max_input_size must be 0"
        assert self.rpu_config.mapping.max_output_size == 0, "max_output_size must be 0"
        assert (
            self.rpu_config.forward.bound_management == BoundManagementType.NONE
        ), "bound_management must be NONE"

        self.is_cuda = False
        self.device = torch.device("cpu")

        # initialize the input ranges
        self.input_range = None
        ir_params = self.rpu_config.pre_post.input_range
        if ir_params.enable:
            self.input_range_update_idx = torch.nn.Parameter(torch.tensor(0., requires_grad=False))
            if ir_params.learn_input_range:
                self.input_range = torch.nn.Parameter(
                    torch.full(
                        (2, self.n_experts),
                        ir_params.init_value,
                        requires_grad=True,
                    )
                )
            else:
                input_range = torch.full(
                    (2, self.n_experts),
                    ir_params.init_value,
                    requires_grad=False,
                )
                if hasattr(self, "input_range") and self.input_range is None:
                    delattr(self, "input_range")
                self.register_buffer("input_range", input_range)  # type: ignore

        self.set_weights(
            expert_sel=self.expert_sel,
            keys=self.keys,
            values=self.values,
            bias=self.bias,
            o_bias=self.o_bias,
        )

    @classmethod
    def from_digital(
        cls,
        module: SigmaMoELayer,
        rpu_config: RPUConfigBase,
        tile_module_class: Optional[Type] = None,
    ) -> "AnalogSigmaMoELayerAIHWKIT":
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
        expert_sel: Tensor,
        keys: Tensor,
        values: Tensor,
        bias: Tensor | None = None,
        o_bias: Tensor | None = None,
    ) -> None:
        self.expert_sel = expert_sel
        self.keys = keys
        self.values = values
        self.analog_ctx = AnalogContext(self)
        self.analog_ctx.use_torch_update = True
        self.bias = bias
        self.o_bias = o_bias

    @staticmethod
    def modify_weight(
        inp_weight: Tensor,
        modifier: WeightModifierParameter,
        remap_type: WeightRemapType,
    ):
        """
        Apply noise injection to weights.

        Args:
            inp_weight (Tensor): Weight to be modified
            modifier (WeightModifierParameter): Parameters
            remap_type (WeightRemapType): What kind of remapping we assume

        Returns:
            Tensor: Modified weight
        """
        # pylint: disable=unused-argument

        per_batch_sample = modifier.per_batch_sample
        if per_batch_sample:
            raise MoEConifgError("per_batch_sample not implemented")

        if modifier.type in [WeightModifierType.NONE, WeightModifierType.COPY]:
            return inp_weight

        if modifier.type == WeightModifierType.MULT_NORMAL:
            raise MoEConifgError("MULT_NORMAL not implemented")

        if remap_type == WeightRemapType.CHANNELWISE_SYMMETRIC:
            assumed_wmax = inp_weight.abs().amax(1)
            assumed_wmax = assumed_wmax.unsqueeze(1)  # [n_experts, 1, n_columns]
        elif remap_type == WeightRemapType.LAYERWISE_SYMMETRIC:
            # [n_experts, 1, 1]
            assumed_wmax = (
                inp_weight.view(inp_weight.size(0), -1).abs().amax(-1).view(-1, 1, 1)
            )
        else:
            raise MoEConifgError(f"Weight remap type {remap_type} not supported")

        if modifier.type == WeightModifierType.DISCRETIZE:
            raise MoEConifgError("DISCRETIZE not implemented")
        elif modifier.type == WeightModifierType.ADD_NORMAL:
            with torch.no_grad():
                noise = (
                    modifier.std_dev
                    * assumed_wmax
                    * torch.randn(inp_weight.shape, device=inp_weight.device)
                )
            out_weight = inp_weight.clone() + noise
        else:
            raise MoEConifgError(f"Weight modifier {modifier} not supported")
        return out_weight

    def cvmm_wrapper(self, inputs: Tensor, sel_indices: CVMMSel, weights: Tensor):
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
                        inputs,
                        sel_indices.sel_index,
                        sel_indices.sel,
                        self.n_experts
                    )
                    if (stds > 0.0).any():
                        self.input_range.data[ir_idx] = (
                            self.input_range.data[ir_idx][stds > 0] * idx + ir_params.init_std_alpha * stds[stds > 0]
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

        if self.rpu_config.forward.is_perfect:
            inp_res = -1  # no quantization
            # TODO also other things like out noise should be disabled

        if self.training or self.rpu_config.modifier.enable_during_test:
            # weight noise injection
            weights = AnalogSigmaMoELayerAIHWKIT.modify_weight(
                weights, self.rpu_config.modifier, self.rpu_config.remap.type
            )

        out_noise = None
        if (
            self.training
            and not self.rpu_config.forward.is_perfect
            and self.rpu_config.forward.out_noise > 0
        ):
            # [bsz, seq_len, top-k, d_out]
            out_noise = torch.randn(
                (*inputs.shape[:2], self.n_heads, weights.shape[-1]),
                device=inputs.device,
            )
            # the inputs into the MVM will be in [-1, 1] range, but the weights are not normalized
            # so we need to scale the noise by the abs max
            if self.rpu_config.remap.type in [
                WeightRemapType.CHANNELWISE_SYMMETRIC,
                WeightRemapType.NONE,
            ]:
                # scale by abs_max of weight channels
                assumed_wmax = weights.abs().amax(1)
            elif self.rpu_config.remap.type == WeightRemapType.LAYERWISE_SYMMETRIC:
                # scale by abs_max of weight layer
                assumed_wmax = (
                    weights.view(weights.size(0), -1).abs().amax(-1).unsqueeze(-1)
                )
            else:
                raise MoEConifgError(
                    f"Weight remap type {self.rpu_config.remap.type} not supported"
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
            weights,
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

    @torch.no_grad()
    def clip_weights(self, clip: WeightClipParameter) -> None:
        """Clip the weights.

        Args:
            clip: parameters specifying the clipping methof and type.

        Raises:
            NotImplementedError: For unsupported WeightClipTypes
            ConfigError: If unknown WeightClipType used.
        """
        if clip.type == WeightClipType.FIXED_VALUE:
            self.keys.data = torch.clamp(self.keys, -clip.fixed_value, clip.fixed_value)
            self.values.data = torch.clamp(
                self.values, -clip.fixed_value, clip.fixed_value
            )
        elif clip.type == WeightClipType.LAYER_GAUSSIAN:
            alpha_keys = self.keys.std((1, 2), keepdims=True) * clip.sigma
            alpha_values = self.values.std((1, 2), keepdims=True) * clip.sigma
            if clip.fixed_value > 0:
                alpha_keys = min(clip.fixed_value, alpha_keys)
                alpha_values = min(clip.fixed_value, alpha_values)
            self.keys.data = torch.clamp(self.keys, -alpha_keys, alpha_keys)
            self.values.data = torch.clamp(self.values, -alpha_values, alpha_values)
        elif clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise NotImplementedError("AVERAGE_CHANNEL_MAX not implemented")
        else:
            raise MoEConifgError(f"Unknown clip type {clip.type}")

    def get_dtype(self) -> dtype:
        return self.keys.dtype

    def to(self, *args, **kwargs) -> "AnalogSigmaMoELayerAIHWKIT":
        # pylint: disable=invalid-name
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
        elif len(args) > 0 and not isinstance(args[0], (Tensor, dtype)):
            device = torch_device(args[0])

        if device is not None:
            device = torch_device(device)
            self.device = device
            if device.type == "cuda":
                self.is_cuda = True
            elif device.type == "cpu":
                self.is_cuda = False
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[torch.device] = None) -> "AnalogSigmaMoELayerAIHWKIT":
        self.is_cuda = True
        self.device = device
        return super().cuda(device=device)

    def cpu(self) -> "AnalogSigmaMoELayerAIHWKIT":
        self.is_cuda = False
        self.device = torch.device("cpu")
        return super().cpu()

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        strict: bool = True,
        load_rpu_config: bool | None = None,
        strict_rpu_config_check: bool | None = None,
    ) -> NamedTuple:
        analog_tile_state = state_dict.pop("analog_tile_state", None)
        if load_rpu_config:
            self.rpu_config = analog_tile_state["rpu_config"]
        if strict_rpu_config_check:
            rpu_compat, compat_msg = self.compatible_with(
                analog_tile_state["rpu_config"]
            )
            if not rpu_compat:
                raise MoEConifgError(compat_msg)
        return super().load_state_dict(state_dict, strict)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "analog_tile_state"] = {"rpu_config": self.rpu_config}

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
        analog_tile_state = state_dict.pop(prefix + "analog_tile_state", None)
        if analog_tile_state is not None:
            self.rpu_config = analog_tile_state["rpu_config"]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compatible_with(self, rpu_config: RPUConfigBase) -> Tuple[bool, Optional[str]]:
        """Checks whether current `RPUConfig` is compatible with given
        one.

        Args:
            rpu_config: New `RPUConfig` to check against

        Returns:
            success: Whether the given `RPUConfig` is compatible
            msg: Error message if not
        """
        if not isinstance(self.rpu_config, type(rpu_config)) and not isinstance(
            rpu_config, type(self.rpu_config)
        ):
            return False, (
                "RPU config mismatch: "
                "Cannot replace "
                f"{rpu_config.__class__.__name__} "
                f"with {self.rpu_config.__class__.__name__}"
            )
        return True, None

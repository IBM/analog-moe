from typing import Optional
import torch
from dataclasses import dataclass
import triton
import triton.language as tl

from aihwkit.simulator.configs import InputRangeParameter, NoiseManagementType
from aihwkit.simulator.parameters import IOParameters

# Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
# Based on https://github.com/RobertCsordas/moe_layer

@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    out_index: Optional[torch.Tensor] = None
    reduction_weight: Optional[torch.Tensor] = None

    def clone(self) -> 'CVMMSel':
        return CVMMSel(self.raw_sel, self.sel, self.sel_index, self.out_index, self.reduction_weight)


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()
    return CVMMSel(sel, ssel.view_as(sel), sel_index, None)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'float32', 'allow_tf32']
)
@triton.jit
def cvmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr, input_ranges_ptr, abs_max_ptr, out_noise_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bo, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_out_noise_m, stride_out_noise_n,
    stride_index, stride_sel, stride_input_ranges, stride_out_index, stride_abs_max,
    input_ranges_is_none: tl.constexpr,
    abs_max_is_none: tl.constexpr,
    out_noise_is_none: tl.constexpr,
    out_index_is_none: tl.constexpr,
    inp_res: tl.constexpr, # is -1 if is_fp else is some number > 1
    is_fp: tl.constexpr, # is_fp is True if the input should not be rounded
    float32: tl.constexpr, allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = first_pid_m + (pid % group_size_m)

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
    sel_all = tl.load(sel_ptr + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

    # sel_all could be
    # [0, 0, 0, ..., 1, 1] with the length of this vector being BLOCK_SIZE_M
    # in this case, sel_first = 0, sel_last = 1
    # so matrix_id will be in [0, 1]

    for matrix_id in range(sel_first, sel_last + 1):
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N


        # remap_offs_am would be something like [0, 2, 3, ..., 12, 15]
        # they represent the tokens that are routed to [0, 0, 0, ..., 1, 1]
        # for this round, we only want to save the ones corresponding to expert number matrix_id
        # so we do a comparison with sel_all in the end.
        remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

        if not input_ranges_is_none:
            # load the input ranges for these tokens
            # this is a vector that is of size [BLOCK_SIZE_M] and contains the input ranges of the
            # corresponding experts, i.e. [1.2, 1.2, 1.2, ..., 0.7, 0.7] (sticking to the values above)
            input_ranges_am = tl.load(input_ranges_ptr + stride_input_ranges * offs_am)
            input_ranges_am = input_ranges_am[:, None]

        # Create offset pointers
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # a_ptrs now represents a chunk of size [BLOCK_SIZE_M, BLOCK_SIZE_K] of tokens (or part of tokens)
        # that mostly will be routed to the same expert. Some should be routed to another expert and
        # we calculate it wrongly, but we mask this out.
        a_ptrs = a_ptr + (remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

        if not abs_max_is_none:
            # load the abs_max for these tokens. abs_max is not ordered to the indices, so we have to
            # index it correctly
            abs_max_am = tl.load(abs_max_ptr + stride_abs_max * remap_offs_am)
            abs_max_am = abs_max_am[:, None]

        # b_ptrs now represents a chunk of size [BLOCK_SIZE_K, BLOCK_SIZE_N] of the expert matrix.
        # the expert number is matrix_id. Each expert is of size [K, N], but this block is only [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # the result will be a [BLOCK_SIZE_M, BLOCK_SIZE_N] matrix so we need to iterate over some blocks of K and accumulate.
        b_ptrs = b_ptr + matrix_id * stride_bo + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # this will store the block result
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # the computation of this block happens here
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

            # here, `a` must be scaled and quantized accordingly
            if not input_ranges_is_none:
                # clip between the input ranges
                over_ir_mask = (a > input_ranges_am)
                under_ir_mask = (a < -input_ranges_am)

                a = tl.where(over_ir_mask, input_ranges_am, a)
                a = tl.where(under_ir_mask, -input_ranges_am, a)

                # scale with input ranges
                a = a / input_ranges_am

            if not abs_max_is_none:
                # we need to scale the input_tokens by the abs_max of that token
                a = a / abs_max_am

            if not is_fp:
                # quantize
                tl.device_assert(tl.max(tl.abs(a)) <= 1.0, "max abs is bigger 1.0")
                a = a / inp_res

                # rounding that mirrors rounding beahaviour of torch.round
                a = tl.extra.cuda.libdevice.rint(a)
                # a_rounded = tl.math.round(a)
                # mask_point_five = tl.abs(tl.abs(a - a_rounded) - 0.5) < 1e-4
                # a = tl.where(mask_point_five & (a_rounded % 2 == 1) & (a_rounded > 0), a_rounded - 1, a_rounded)
                # a = tl.where(mask_point_five & (a_rounded % 2 == 1) & (a_rounded < 0), a_rounded + 1, a)

                a = a * inp_res

            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # we accumulate along the K dimension

            if not float32:
                a = a.to(tl.float16)
                b = b.to(tl.float16)

            accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            # advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk


        if not float32:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        # this is where we now write the output into the output matrix.
        if out_index_is_none:
            # if this is the down projection, the input was already [bsz * seq_len * top-k, d_ff]
            # so the remap_offs_cm are in the range of [0, bsz * seq_len * top-k)
            remap_offs_cm = remap_offs_am
        else:
            # if this is the up projection, the input is just [bsz * seq_len, d_model]
            # but the output is [bsz * seq_len * top-k, d_ff]
            # so we need to use the indices that range from [0, bsz * seq_len * top-k)
            # which are stored in out_index_ptr
            remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
        
        # we don't want to store the results where the tokens should have been routed to a different
        # expert, so we mask it out with sel_all[:, None] == matrix_id
        # sel_all is this vector [0, 0, 0, ..., 1, 1] and matrix id is an integer in this case between [0, 1]
        c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
        
        if not out_noise_is_none:
            # add the out_noise
            out_noise = tl.load(
                out_noise_ptr + stride_out_noise_m * remap_offs_cm[:, None] + stride_out_noise_n * offs_cn[None, :],
                mask=c_mask,
                other=0.0
            )
            c = c + out_noise

        if not abs_max_is_none:
            c = c * abs_max_am

        # scale down if input_ranges is not None
        if not input_ranges_is_none:
            c = c * input_ranges_am

        tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'float32_out', 'allow_tf32', 'op_float16'], reset_to_zero = ['c_ptr']
)
@triton.jit
def cvmm_backward_kernel3(
    # x,   grads, out,   sel_index, sel,     out_index,     input_ranges
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr, input_ranges_ptr,
    # d_exp, d_model, bsz*seq_len*k
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_co, stride_cm, stride_cn,
    stride_index, stride_sel, stride_input_ranges, stride_out_index,
    input_ranges_is_none: tl.constexpr,
    out_index_is_none: tl.constexpr,
    float32_out: tl.constexpr, allow_tf32: tl.constexpr, op_float16: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, K_BLOCKS: tl.constexpr
):
    # this is a grid, so pid can be the same for two kernel instances
    pid = tl.program_id(axis=0)
    k_block_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # since this is a 2D grid launch, pid_m and pid_n can be the same for two kernel instances
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn

    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS # 0
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1 # min(0 + 16 * 32 = 512, 200) -> 199

    # K is the number of tokens, so bsz*seq_len*k
    # our 2D-grid is over M*N and K
    # sel block_start_index to block_end_index can hold e.g. [0, 0, 0, ..., 3, 3]
    first_mat = tl.load(sel_ptr + stride_sel * block_start_index) # e.g 0
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index) # e.g. 3

    for matrix_index in range(first_mat, last_mat + 1): # e.g. over [0,1,2,3]
        # this will hold part of the gradient. M: d_exp N: d_model (in the down_projection case)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # this will be explained below
        start_i = block_start_index
        end_i = block_end_index + 1
        while start_i < end_i:
            middle = (start_i + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix < matrix_index:
                start_i = middle + 1
            else:
                end_i = middle
        start_i2 = start_i
        end_i = block_end_index + 1
        while start_i2 < end_i:
            middle = (start_i2 + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix <= matrix_index:
                start_i2 = middle + 1
            else:
                end_i = middle

        end_i = start_i2
        count = end_i - start_i
        block_mem_indices_f_base = start_i  + tl.arange(0, BLOCK_SIZE_K)

        # the above code finds a consecutive block where the tokens
        # get routed to the same expert.
        # sel[start_i:end_i] will be something like [0, 0, 0, ..., 0]
        # of course, sometimes it will be empty, so count will be 0
        if count > 0:
            for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                # let's say we found a consecutive block of 100 tokens that are routed to the same expert
                # now, we only want to operate on sub-vectors of size BLOCK_SIZE_K
                # so we need tl.cdiv(100, 16) = 7 iterations
                
                # here we actually get the indices, could be [45, 46, ... 45+BLOCK_SIZE_K]
                block_mem_indices_f = block_mem_indices_f_base + k * BLOCK_SIZE_K
                # but the last index could overshoot and we could access out of bounds
                # so we have to use the modulo of how big K can actually be
                block_mem_indices = block_mem_indices_f % K
                # here load the actual tokens (using index_ptr as the offset and not sel_ptr)
                # sel only tells you the expert, but doesn't tell you where the token is in the `a` matrix
                a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                if out_index_is_none:
                    b_index = a_index
                else:
                    # the output in the forward of the keys uses this mapping, so we also
                    # need to use it here for dL/dp1
                    b_index = tl.load(out_index_ptr + stride_out_index * block_mem_indices)
                sel_ok = block_mem_indices_f < end_i

                a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk
                
                # load the corresponding input ranges
                input_ranges_ptrs = input_ranges_ptr + stride_input_ranges * block_mem_indices

                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)
                
                if not input_ranges_is_none:
                    input_ranges = tl.load(input_ranges_ptrs, mask=sel_ok)
                    input_ranges = input_ranges[None, :]
                    # clip between the input ranges
                    over_ir_mask = sel_ok & (a > input_ranges)
                    under_ir_mask = sel_ok & (a < -input_ranges)

                    a = tl.where(over_ir_mask, input_ranges, a)
                    a = tl.where(under_ir_mask, -input_ranges, a)

                if op_float16:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)

                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            if float32_out:
                c = accumulator
            else:
                c = accumulator.to(tl.float16)

            # pid_m and pid_n can be the same for two different kernel instances, which is why
            # we need to perform an atomic add
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_co * matrix_index + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            
            # we could be writing to the same memory location (since we are spawning across K)
            # this means process 1 could operate on [0,0,...,0] and process two on [0,0,0,..1]
            # so on tokens that map to the same expert.
            tl.atomic_add(c_ptrs, c, mask=c_mask)


def cvmm_std(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    n_experts: int,
):
    """
    Calculates the per-expert std of the input tokens.

    Args:
        x (torch.Tensor): _description_
        sel_index (torch.Tensor): _description_
        sel (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    # collapses all of the dimensions except the last one
    x = x.flatten(end_dim=-2)
    x_sum = x.sum(dim=-1) / x.size(-1) # collapse it to number of tokens
    sel = sel.flatten() # len(sel) = bsz * seq_len * top-k
    per_expert_sum = torch.zeros((n_experts, ), device=x.device, dtype=x.dtype)
    per_expert_count1 = torch.zeros((n_experts, ), device=x.device, dtype=x.dtype)
    center_and_square = torch.zeros((n_experts, ), device=x.device, dtype=x.dtype)
    per_expert_count2 = torch.zeros((n_experts, ), device=x.device, dtype=x.dtype)

    M = sel.size(0)
    N = x.size(-1)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
    )
    cvmm_mean_kernel[grid](
        x_sum, per_expert_sum, per_expert_count1, sel_index, sel, M,
    )
    per_expert_mean = per_expert_sum / per_expert_count1

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    cvmm_center_and_square_kernel[grid](
        x, center_and_square, per_expert_count2, per_expert_mean, x.stride(0), x.stride(1), sel_index, sel, M, N
    )
    return torch.sqrt(center_and_square / (per_expert_count2-1))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['M','N'], reset_to_zero=['out_ptr', 'count_ptr']
)
@triton.jit
def cvmm_center_and_square_kernel(
    x_ptr, out_ptr, count_ptr, per_expert_mean_ptr,
    stride_xm, stride_xn,
    sel_index_ptr, sel_ptr, M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
    sel_all = tl.load(sel_ptr + ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

    for matrix_id in range(sel_first, sel_last + 1):
        mean = tl.load(per_expert_mean_ptr + matrix_id)
        offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        remap_offs_xm = tl.load(sel_index_ptr + offs_xm)
        belonging_to_this_matrix = sel_all == matrix_id
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        x_ptrs = x_ptr + (remap_offs_xm[:, None] * stride_xm + offs_n[None, :] * stride_xn)

        # the computation of this block happens here
        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            n_mask = offs_n[None, :] < N - n * BLOCK_SIZE_N
            x = tl.load(x_ptrs, mask=n_mask, other=0.0)
            x_minus_mean = (x - mean)
            x_minus_mean_sq = x_minus_mean * x_minus_mean
            mask = belonging_to_this_matrix[:, None] & n_mask
            x_minus_mean_sq = tl.where(mask, x_minus_mean_sq, 0.0)
            x_ptrs += BLOCK_SIZE_N * stride_xn
            tl.atomic_add(out_ptr + matrix_id, tl.sum(x_minus_mean_sq))
            tl.atomic_add(count_ptr + matrix_id, tl.sum(tl.where(mask, 1., 0.)))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
    ],
    key=['M'], reset_to_zero=['per_expert_sum_ptr', 'per_expert_count_ptr']
)
@triton.jit
def cvmm_mean_kernel(
    x_sum_ptr, per_expert_sum_ptr, per_expert_count_ptr,
    sel_index_ptr, sel_ptr, M,
    BLOCK_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1))
    sel_all = tl.load(sel_ptr + ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

    for matrix_id in range(sel_first, sel_last + 1):
        offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        remap_offs_xm = tl.load(sel_index_ptr + (offs_xm % M))
        belonging_to_this_matrix = (sel_all == matrix_id) & (offs_xm < M)
        tl.atomic_add(
            per_expert_sum_ptr + matrix_id,
            tl.sum(tl.load(x_sum_ptr + remap_offs_xm, mask=belonging_to_this_matrix, other=0.0))
        )
        count = tl.where(belonging_to_this_matrix, 1., 0.)
        sum_count = tl.sum(count)
        tl.atomic_add(
            per_expert_count_ptr + matrix_id,
            sum_count
        )


torch.library.define("mylib::cvmm_triton_quantized", "(Tensor x, Tensor sel_index, Tensor sel, Tensor keys, Tensor input_ranges, float inp_res, ScalarType out_dtype, Tensor out_index, Tensor abs_max, Tensor out_noise) -> Tensor")
@torch.library.impl("mylib::cvmm_triton_quantized", "default")
def cvmm_triton_quantized(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    keys: torch.Tensor,
    input_ranges: torch.Tensor,
    inp_res: float,
    out_dtype: torch.dtype,
    out_index: torch.Tensor,
    abs_max: torch.Tensor,
    out_noise: torch.Tensor,
):
    """
    TODO

    Args:
        x (torch.Tensor): Shape [bsz, seq_len, d_model] or [bsz, seq_len, top-k, d_ff]
        sel_index (torch.Tensor): Shape [bsz * seq_len * top-k]
        sel (torch.Tensor): Shape [bsz, seq_len, top-k]
        keys (torch.Tensor): Shape [n_experts, d_model, d_ff] or [n_experts, d_ff, d_model]
        out_dtype (torch.dtype): Type of output.
        out_index (torch.Tensor): Shape [bsz * seq_len * top-k]
        abs_max (torch.Tensor): Shape [bsz, seq_len, (maybe top-k), 1]
        out_noise (torch.Tensor): Shape [bsz, seq_len, top-k, d_ff/d_model]

    Returns:
        _type_: _description_
    """
    # collapses all of the dimensions except the last one
    x = x.flatten(end_dim=-2)
    assert x.shape[-1] == keys.shape[1]

    sel_shape = sel.shape
    sel = sel.flatten()

    input_ranges = input_ranges.flatten()

    abs_max = abs_max.flatten()

    M = sel.shape[0]
    O, K, N = keys.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    # out = torch.zeros((M, N), device=x.device, dtype=out_dtype)
    # 1D launch kernel where each block gets its own program.

    # expected_m_per_matrix = int(math.ceil(M / O * 1.5))
    # expected_m_per_matrix = M

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    input_ranges_is_none = False
    if input_ranges.numel() == 1 and input_ranges == -1:
        input_ranges_is_none = True

    abs_max_is_none = False
    if abs_max.numel() == 1 and abs_max == -1:
        abs_max_is_none = True

    out_noise_is_none = False
    if out_noise.numel() == 1 and out_noise == -1:
        out_noise_is_none = True
    else:
        out_noise = out_noise.flatten(end_dim=-2)
        assert out_noise.shape[1] == keys.shape[2]

    import os
    skip_rounding = os.environ.get("_AIHWKIT_NO_ROUNDING", False)
    if skip_rounding:
        assert os.environ.get("AIHWKIT_TESTING", False),\
            "Skipping rounding but AIHWKIT_TESTING not set"
        inp_res = -1

    cvmm_kernel[grid](
        x, keys, out, sel_index, sel, out_index, input_ranges, abs_max, out_noise,
        M, N, K,
        x.stride(0), x.stride(1),
        keys.stride(0), keys.stride(1), keys.stride(2),
        out.stride(0), out.stride(1),
        0 if out_noise_is_none else out_noise.stride(0),
        0 if out_noise_is_none else out_noise.stride(1),
        sel_index.stride(0), sel.stride(0),
        0 if input_ranges_is_none else input_ranges.stride(0),
        0 if out_index_is_none else out_index.stride(0),
        0 if abs_max_is_none else abs_max.stride(0),
        input_ranges_is_none=input_ranges_is_none,
        abs_max_is_none=abs_max_is_none,
        out_noise_is_none=out_noise_is_none,
        out_index_is_none=out_index_is_none,
        inp_res=inp_res,
        is_fp=inp_res == -1,
        float32=out.dtype==torch.float32, allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
    )

    return out.view(*sel_shape, N)


@torch.library.impl_abstract("mylib::cvmm_triton_quantized", cvmm_triton_quantized)
def cvmm_triton_quantized_abstract(x, sel_idx, sel, keys, input_ranges, inp_res, out_dtype, out_index, abs_max, out_noise):
    sel_shape = sel.shape
    sel = sel.flatten()
    M = sel.shape[0]
    O, K, N = keys.shape
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    sel_shape = sel.shape
    return out.view(*sel_shape, N)


def cvmm_triton_backward(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    grads: torch.Tensor,
    input_ranges: torch.Tensor,
    n_experts: int,
    key_dtype: torch.dtype,
    op_float16: bool,
    out_index: torch.Tensor
):
    # values part:
    # forward (:p2) [bsz,seq_len,k,d_exp] -> (values) [bsz,seq_len,k,d_model] (:z) -> (w_red) [bsz,seq_len,d_model] (:y) -> ...
    # x: p2
    # flatten to [bsz*seq_len*k,d_exp]
    x = x.flatten(end_dim=-2)
    # shape is [d_exp,bsz_seq_len*k]
    x = x.transpose(0, 1)
    # grads: dL/dz : [bsz,seq_len,k,d_model] -> [bsz*seq_len*k,d_model]
    grads = grads.flatten(end_dim=-2)
    # sel: [bsz,seq_len,k] -> [bsz*seq_len*k], same as sel_index
    sel = sel.flatten()
    # also flatten the input ranges
    input_ranges = input_ranges.flatten()
    M, _ = x.shape
    K, N = grads.shape
    out = torch.zeros((n_experts, M, N), device=x.device, dtype=key_dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K'] * META['K_BLOCKS'])
    )
    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    input_ranges_is_none = False
    if input_ranges.numel() == 1 and input_ranges == -1:
        input_ranges_is_none = True

    cvmm_backward_kernel3[grid](
        x, grads, out, sel_index, sel, out_index, input_ranges,
        M, N, K,
        x.stride(0), x.stride(1),
        grads.stride(0), grads.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sel_index.stride(0), sel.stride(0), input_ranges.stride(0), 0 if out_index_is_none else out_index.stride(0),
        input_ranges_is_none=input_ranges_is_none,
        out_index_is_none=out_index_is_none,
        float32_out=out.dtype == torch.float32,
        op_float16=op_float16,
        allow_tf32=False #torch.backends.cuda.matmul.allow_tf32
    )
    return out


class CVMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        inp_res: float,
        broadcasted_input_ranges: Optional[torch.Tensor] = None,
        out_index: Optional[torch.Tensor] = None,
        reduction_weight: Optional[torch.Tensor] = None,
        out_noise: Optional[torch.Tensor] = None,
        ir_params: Optional[InputRangeParameter] = None,
        io_params: Optional[IOParameters] = None,
    ):
        # ctx.save_for_backward(x, keys, sel, sel_index, out_index, reduction_weight, input_ranges)
        out_type = torch.float16 if torch.is_autocast_enabled() else x.dtype
        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1).cuda()

        input_ranges_is_none = False
        if broadcasted_input_ranges is None:
            input_ranges_is_none = True
            broadcasted_input_ranges = torch.tensor(-1).cuda()

        is_aihwkit_lightning = not hasattr(io_params, "is_perfect")

        if is_aihwkit_lightning:
            if ir_params.enable:
                abs_max = torch.tensor(-1).cuda()
            else:
                abs_max = x.abs().amax(-1, keepdim=True)
        else:
            if io_params is not None and not io_params.is_perfect and io_params.noise_management == NoiseManagementType.ABS_MAX:
                abs_max = x.abs().amax(-1, keepdim=True)
            else:
                abs_max = torch.tensor(-1).cuda()

        if out_noise is None:
            out_noise = torch.tensor(-1).cuda()
        
        res = torch.ops.mylib.cvmm_triton_quantized(
            x,
            sel_index,
            sel,
            keys,
            broadcasted_input_ranges,
            inp_res,
            out_type,
            out_index,
            abs_max,
            out_noise
        )

        if reduction_weight is not None:
            res_into_reduction = res.view(*reduction_weight.shape, res.shape[-1])
            res = (reduction_weight.unsqueeze(-2).type_as(res) @ res_into_reduction).squeeze(-2)
        else:
            res_into_reduction = None

        ctx.save_for_backward(x, keys, sel, sel_index, None if out_index_is_none else out_index, reduction_weight, None if input_ranges_is_none else broadcasted_input_ranges, res_into_reduction)
        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.is_autocast = torch.is_autocast_enabled()
        # control for input_range gradients
        ctx.ir_params = ir_params
        return res

    @staticmethod
    def backward(ctx, grad_output):

        # total Forward: x: [sz,seq_len,d_model], -> (keys) [bsz,seq_len,k,d_exp] (:p1)|  (torch)  |(:p2) [bsz,seq_len,k,d_exp] -> (values) [bsz,seq_len,k,d_model] (:z) -> (w_red) [bsz,seq_len,d_model] (:y) -> ...

        # NOTE uncomment this is for debugging using e.g. VSCode
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        x, keys, sel, sel_index, out_index, reduction_weight, broadcasted_input_ranges, res_into_reduction = ctx.saved_tensors
        keys_dt = keys

        ir_params: InputRangeParameter = ctx.ir_params
        
        if reduction_weight is not None:
            # forward (:p2) [bsz,seq_len,k,d_exp] -> (values) [bsz,seq_len,k,d_model] (:z) -> (w_red) [bsz,seq_len,d_model] (:y) -> ...
            # backward: dL/dy [bsz,seq_len,d_model] is given, we can compute dL/dz [bsz,seq_len,k,d_model] by multiplying with w_red transpose:
            # dL/dz = w_red^T : [bsz, seq_len, k, 1] * dL/dy : [bsz,seq_len,1,d_model] -> [bsz,seq_len,k,d_model]
            grad_output_w = reduction_weight.unsqueeze(-1).type_as(grad_output) @ grad_output.unsqueeze(-2)
        else:
            # forward: x: [sz,seq_len,d_model], -> (keys) [bsz,seq_len,k,d_exp] (:p1)|
            # we are already further in the backward pass. Here, we already get dL/dp1 so we don't have to do anything.
            grad_output_w  = grad_output

        input_ranges_is_none = False
        if broadcasted_input_ranges is None:
            input_ranges_is_none = True
            broadcasted_input_ranges = torch.tensor(-1).cuda()

        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1).cuda()

        # backward: (values part)
        # compute dL/d values using p2 and dL/dz. p2 has shape [bsz,seq_len,k,d_exp] and dL/dz has shape [bsz,seq_len,k,d_model]
        # it is easier to reshape the inputs and dL/dz to [bsz*seq_len*k, d_exp] and [bsz*seq_len*k, d_model]
        # now, we start with expert 1. lets say we know all the tokens that are routed to expert 1. We can now compute the gradient
        # dL/d value1 = p2^T : [d_exp, num_tokens_routed_to_exp1] @ dL/dz : [num_tokens_routed_to_exp1, d_model]
        # in this case, the out_index will not be populated. We use the sel_index instead to know which tokens are routed to which expert.
        # the part for the keys (up-proj) is similar
        grad_w = cvmm_triton_backward(
            x, # p2 in the case of the values backward pass and actual x in the case for the keys backward pass
            sel_index, # [bsz*seq_len*k] indices from 0 to bsz*seq_len*k-1 (for values bw) and bsz*seq_len-1 (for keys bw)
            # that represent memory locations of the tokens that get first routed to expert 1, then expert 2, etc.
            sel, # ordered indices from 0 to n_experts-1 that represent the expert number that the token should be routed to
            grad_output_w, # dL/dz for the values, and dL/dp1 for the keys
            broadcasted_input_ranges, # input ranges
            keys_dt.shape[0], 
            ctx.keys_type,
            ctx.is_autocast,
            out_index=out_index # for the values backward, this is None, for the keys backward this is the indices from 0 to bsz*seq_len*k-1
        )

        # Backward for input and reduction weight
        grad_w_off = None

        bw_index = sel_index if out_index_is_none else out_index
        bw_index_out = torch.tensor(-1).cuda()
        if reduction_weight is not None:
            # Hack the output indices to emulate repeats
            bw_index_out = bw_index
            bw_index = bw_index // reduction_weight.shape[-1]

        grad_x_full = torch.ops.mylib.cvmm_triton_quantized(
            grad_output, # dL / dp1 for the values backward, dL / dy for the keys backward
            bw_index,
            sel,
            keys_dt.transpose(1,2),
            torch.tensor(-1).cuda(),
            -1, # full precision
            ctx.op_type,
            bw_index_out,
            torch.tensor(-1).cuda(),
            torch.tensor(-1).cuda(),
        )

        if not input_ranges_is_none:
            # how many x did actually clip?
            arg_sorted_indices = torch.argsort(sel_index if out_index_is_none else out_index)
            fl_x_sel_index = x.view(-1, x.size(-1))[sel_index]
            fl_broadcasted_input_ranges = broadcasted_input_ranges.view(-1, 1)
            did_x_not_clip = (fl_x_sel_index.abs() < fl_broadcasted_input_ranges)[arg_sorted_indices]

            # calculate the gradient for the input ranges
            upper_thresh = (fl_x_sel_index >= fl_broadcasted_input_ranges)
            lower_thresh = (fl_x_sel_index <= -fl_broadcasted_input_ranges)
            
            grad_x_full_view_sorted = grad_x_full.view(-1, grad_x_full.size(-1))[sel_index if out_index_is_none else out_index]
            if reduction_weight is not None:
                grad_x_full_view_sorted *= reduction_weight.view(-1, 1)[sel_index]
            
            ir_grad = torch.zeros_like(fl_broadcasted_input_ranges)
            ir_grad += torch.clamp(upper_thresh * grad_x_full_view_sorted, min=None, max=0.0).sum(-1, keepdim=True)
            ir_grad -= torch.clamp(lower_thresh * grad_x_full_view_sorted, min=0.0, max=None).sum(-1, keepdim=True)

            # aihwkit_lightning doesn't have gradient_relative (it's always relative)
            if getattr(ir_params, "gradient_relative", True):
                ir_grad *= fl_broadcasted_input_ranges
                ir_grad *= getattr(ir_params, "gradient_scale", 1.0)

            if ir_params.decay > 0:
                percentage = did_x_not_clip.float().mean()
                ir_grad += (
                    ir_params.decay
                    * fl_broadcasted_input_ranges
                    * (percentage > ir_params.input_min_percentage)
                )

            # d clip(x) / d x
            grad_x_full = grad_x_full * did_x_not_clip.view_as(grad_x_full)


        grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
        if reduction_weight is not None:
            # grad_x_full is the unscaled grad. For the input, we have to scale it, for the reduction weight,
            # we have to compute dot products with the input.
            grad_x = (reduction_weight.view(*grad_x_full.shape[:-1]).unsqueeze(-2).type_as(grad_x_full) @ grad_x_full).squeeze(-2)
            grad_w_off = (res_into_reduction.type_as(reduction_weight) @ grad_output.unsqueeze(-1).type_as(reduction_weight)).squeeze(-1).view_as(reduction_weight)
        elif grad_x_full.shape[-2] != 1:
            grad_x = grad_x_full.sum(-2)
        else:
            grad_x = grad_x_full

        grad_x = grad_x.view_as(x)

        return grad_x, None, None, grad_w, None, None if input_ranges_is_none else ir_grad.view_as(broadcasted_input_ranges), None, grad_w_off, None, None, None
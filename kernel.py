import torch

import triton
import triton.language as tl

# GEMM 1 kernel
@triton.jit
def moe_gemm1_kernel(
    # Hidden states weights, scales, strides
    # A: [Tk, H], A_scale: [H/128, T]
    A_ptr: tl.pointer_type, A_scale_ptr: tl.pointer_type,
    a_stride_t: tl.constexpr, a_stride_h: tl.constexpr,
    a_scale_stride_h: tl.constexpr, a_scale_stride_t: tl.constexpr,
    # Expert weights, scales, strides
    # W: [E, 2I, H], W_scale: [E, 2I/128, H/128]
    W_ptr: tl.pointer_type, W_scale_ptr: tl.pointer_type,
    w_stride_e: tl.constexpr, w_stride_i: tl.constexpr, w_stride_h: tl.constexpr,
    w_scale_stride_e: tl.constexpr, w_scale_stride_i: tl.constexpr, w_scale_stride_h: tl.constexpr,
    # Output placeholder, scales, strides
    # O: [Tk, I], O_scale: [I/BLOCK_N, # tiles]
    O_ptr: tl.pointer_type, O_scale_ptr: tl.pointer_type,
    out_stride_t: tl.constexpr, out_stride_i: tl.constexpr,
    out_scale_stride_i: tl.constexpr, out_scale_stride_t: tl.constexpr,
    # Expert position calculation helpers
    sorted_token_ids_ptr: tl.pointer_type,
    expert_start_ptr: tl.pointer_type,
    tile_expert_ids_ptr: tl.pointer_type,
    tile_local_m_ptr: tl.pointer_type,
    # Compile time constants
    T: tl.constexpr, H: tl.constexpr, I: tl.constexpr, I2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, TOTAL_TILES: tl.constexpr
):
    """Fused first GEMM + SwiGLU for MoE.

    W shape: [E, I2, H] where I2 = 2*I.
    Columns [0, I) are the gate (X1), columns [I, I2) are the value (X2).
    Output shape: [total_tokens, I] (half the width, after SwiGLU).

    Grid: (num_m_tiles, num_n_tiles) where num_n_tiles covers I (not I2).
    Each block computes BLOCK_N columns of the final I-wide output.

    We require that BLOCK_K, BLOCK_N <= 128 so that we only have to apply one scale
    """
    pid = tl.program_id(axis=0)
    num_pid_m = TOTAL_TILES
    num_pid_n = tl.cdiv(I, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    tile_idx = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert  = tl.load(tile_expert_ids_ptr + tile_idx).to(tl.int32)
    local_m = tl.load(tile_local_m_ptr    + tile_idx).to(tl.int32)

    tok_start = tl.load(expert_start_ptr + expert).to(tl.int32)
    tok_end   = tl.load(expert_start_ptr + expert + 1).to(tl.int32)
    num_toks  = tok_end - tok_start

    m_offs   = local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok_mask = m_offs < num_toks
    packed_ids = tok_start + m_offs

    # Column offsets into the I-wide output space
    n_offs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
    n_mask = n_offs < I

    # Two accumulators: one for X1 (gate), one for X2 (value)
    acc_x1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_x2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Column offsets into the full I2-wide weight matrix
    # X1 = W[:, 0:I],  X2 = W[:, I:I2]
    n_offs_x1 = n_offs            # columns [0, I)
    n_offs_x2 = n_offs + I        # columns [I, I2)

    for k in range(0, H, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        # Gather rows of A for selected tokens — shared by both halves
        a = tl.load(
            A_ptr + packed_ids[:, None] * a_stride_t + k_offs[None, :],
            mask=tok_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        # Load gate weight tile (X1 half)
        w_x1 = tl.load(
            W_ptr + expert * w_stride_e + n_offs_x1[:, None] * w_stride_i + k_offs[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        # Load value weight tile (X2 half)
        w_x2 = tl.load(
            W_ptr + expert * w_stride_e + n_offs_x2[:, None] * w_stride_i + k_offs[None, :],
            mask=(n_offs_x2 < I2)[:, None] & k_mask[None, :],
            other=0.0
        )

        # Per-token activation scale
        scale_a = tl.load(
            A_scale_ptr + (k // 128) * a_scale_stride_h + packed_ids * a_scale_stride_t,
            mask=tok_mask, other=1.0
        )

        # Weight scales for each half (different I-tile index)
        scale_w_x1 = tl.load(
            W_scale_ptr + expert * w_scale_stride_e
                        + (pid_n * BLOCK_N // 128) * w_scale_stride_i
                        + (k // 128) * w_scale_stride_h
        )
        scale_w_x2 = tl.load(
            W_scale_ptr + expert * w_scale_stride_e
                        + ((pid_n * BLOCK_N + I) // 128) * w_scale_stride_i
                        + (k // 128) * w_scale_stride_h
        )

        # Accumulate both halves
        acc_x1 += tl.dot(a, tl.trans(w_x1), out_dtype=tl.float32) * scale_a[:, None] * scale_w_x1
        acc_x2 += tl.dot(a, tl.trans(w_x2), out_dtype=tl.float32) * scale_a[:, None] * scale_w_x2

    # ---- Fused SwiGLU: silu(X2) * X1 ----
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_x2 = acc_x2 / (1.0 + tl.exp(-acc_x2))
    result = silu_x2 * acc_x1

    max_val = tl.max(tl.abs(result), axis=1) 
    scale = max_val / 448.0
    scale = tl.maximum(scale, 1e-12)
    res_scaled = result / scale[:, None]
    res_scaled = res_scaled.to(tl.float8e4nv)

    # Store I-wide output (not I2)
    out_rows = tok_start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_mask = out_rows < num_toks + tok_start

    tl.store(
        O_ptr + out_rows[:, None] * out_stride_t + n_offs[None, :] * out_stride_i,
        res_scaled,
        mask=out_mask[:, None] & n_mask[None, :]
    )

    tl.store(
        O_scale_ptr + pid_n * out_scale_stride_i + out_rows * out_scale_stride_t,
        scale,
        mask=out_mask
    )

@triton.jit
def moe_gemm2_kernel(
    # Intermediate states weights, scales, strides
    # A: [Tk, I], A_scale: [# tiles, I/BLOCK_I]
    A_ptr: tl.pointer_type, A_scale_ptr: tl.pointer_type,
    a_stride_t: tl.constexpr, a_stride_i: tl.constexpr,
    a_scale_stride_i: tl.constexpr, a_scale_stride_t: tl.constexpr,
    # Expert weights, scales, strides
    # W: [E, H, I], W_scale: [E, H/128, I/128]
    W_ptr: tl.pointer_type, W_scale_ptr: tl.pointer_type,
    w_stride_e: tl.constexpr, w_stride_h: tl.constexpr, w_stride_i: tl.constexpr,
    w_scale_stride_e: tl.constexpr, w_scale_stride_h: tl.constexpr, w_scale_stride_i: tl.constexpr,
    weights_ptr: tl.pointer_type,
    # Output placeholder, strides
    # O: [Tk, H]
    O_ptr: tl.pointer_type,
    out_stride_t: tl.constexpr, out_stride_h: tl.constexpr,
    # Expert position calculation helpers
    sorted_token_ids_ptr: tl.pointer_type,
    expert_start_ptr: tl.pointer_type,
    tile_expert_ids_ptr: tl.pointer_type,
    tile_local_m_ptr: tl.pointer_type,
    # Compile time constants
    T: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_I: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, TOTAL_TILES: tl.constexpr 
):
    """GEMM2
    A[T_total, I] (quantized SwiGLU output) @ W2[E, H, I]^T (FP8E4M3) -> Out[T_total, H]
    We require that BLOCK_K <= BLOCK_I
    """
    pid = tl.program_id(axis=0)
    num_pid_m = TOTAL_TILES
    num_pid_n = tl.cdiv(H, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    tile_idx = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert  = tl.load(tile_expert_ids_ptr + tile_idx).to(tl.int32)
    local_m = tl.load(tile_local_m_ptr    + tile_idx).to(tl.int32)

    tok_start = tl.load(expert_start_ptr + expert).to(tl.int32)
    tok_end   = tl.load(expert_start_ptr + expert + 1).to(tl.int32)
    num_toks  = tok_end - tok_start

    m_offs   = local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ids  = tok_start + m_offs
    row_mask = m_offs < num_toks

    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_offs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)

    for k in range(0, I, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < I

        a = tl.load(
            A_ptr + row_ids[:, None] * a_stride_t + k_offs[None, :] * a_stride_i,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        scale_a = tl.load(
            A_scale_ptr + (k // BLOCK_I) * a_scale_stride_i + row_ids * a_scale_stride_t
        )

        w = tl.load(
            W_ptr + expert * w_stride_e + n_offs[:, None] * w_stride_h + k_offs[None, :] * w_stride_i,
            mask=(n_offs < H)[:, None] & k_mask[None, :],
            other=0.0
        )

        scale_w = tl.load(
            W_scale_ptr + expert * w_scale_stride_e
                        + (pid_n * BLOCK_N // 128) * w_scale_stride_h
                        + (k // 128) * w_scale_stride_i
        )

        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32) * scale_a[:, None] * scale_w

    weights = tl.load(weights_ptr + row_ids, mask=row_mask, other=0.0)
    acc *= weights[:, None]

    tok_mask = m_offs < num_toks
    tok_ids  = tl.load(sorted_token_ids_ptr + tok_start + m_offs,
                       mask=tok_mask, other=0).to(tl.int32)
    out_ptrs = O_ptr + tok_ids[:, None] * out_stride_t + n_offs[None, :] * out_stride_h
    out_mask = tok_mask[:, None] & (n_offs < H)[None, :]

    # Store to Out[T, H]
    tl.atomic_add(
        out_ptrs,
        acc,
        mask=out_mask
    )

@triton.jit
def _count_kernel(
    flat_expert_ptr,          # [M] int32/int64, M = T * TOP_K
    partial_counts_ptr,       # [num_blocks, E_local] int32, zero-initialized
    M,
    lo, E_local,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 1: each block scans BLOCK_SIZE elements, writes a local histogram.

    We emit per-block partial histograms instead of global atomics on
    token_counts, then reduce on the host with a single .sum(0). This keeps
    atomic traffic out of the hot path.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    e = tl.load(flat_expert_ptr + offs, mask=mask, other=-1)
    local_e = e - lo
    is_local = mask & (local_e >= 0) & (local_e < E_local)

    # Register-resident histogram. E_local is small (typ. 8-64), so this fits.
    # Unrolled loop: for each expert id, count matches in this block.
    row_base = pid * E_local
    for i in tl.static_range(0, 1):  # placeholder; real loop below
        pass

    # We do the histogram with a runtime loop over experts. static_range would
    # require E_local to be constexpr; we make it so via the launch.
    for k in range(E_local):
        hit = is_local & (local_e == k)
        cnt = tl.sum(hit.to(tl.int32), axis=0)
        tl.store(partial_counts_ptr + row_base + k, cnt)


@triton.jit
def _scatter_kernel(
    flat_expert_ptr,          # [M]
    flat_weight_ptr,          # [M] float
    expert_start_ptr,         # [E_local] int32 — mutable; we atomic_add into it
    sorted_token_ptr,         # [N] int32 — output
    sorted_weight_ptr,        # [N] float — output
    M,
    TOP_K: tl.constexpr,
    lo,
    E_local: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 2: each lane that hits a local expert atomically claims an output
    slot via atomic_add on expert_start[e], then writes (token, weight) there.

    Note: expert_start is consumed as a cursor. Caller must pass a *copy* of
    the original offsets, or reconstruct offsets afterwards.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    e = tl.load(flat_expert_ptr + offs, mask=mask, other=-1)
    w = tl.load(flat_weight_ptr + offs, mask=mask, other=0.0)
    local_e = e - lo
    is_local = mask & (local_e >= 0) & (local_e < E_local)

    # Token id is flat_idx // TOP_K.
    tok = (offs // TOP_K).to(tl.int32)

    # Atomic claim: returns old value = our output slot.
    # We pass `mask=is_local` so inactive lanes don't participate.
    slot = tl.atomic_add(expert_start_ptr + local_e, 1, mask=is_local, sem="relaxed")

    tl.store(sorted_token_ptr + slot, tok, mask=is_local)
    tl.store(sorted_weight_ptr + slot, w, mask=is_local)


def build_expert_dispatch(topk_idx, topk_weights, local_expert_offset, E_local, TOP_K):
    """Fused Triton MoE dispatch: filter → count → CSR → scatter.

    Notes:
      * Intra-bucket token order is NON-deterministic (atomic race). If you
        need stable order within an expert, sort sorted_token within each
        bucket afterwards, or use the PyTorch version.
      * Returns the same tuple as the reference implementation.
    """
    assert topk_idx.is_cuda and topk_weights.is_cuda
    T = topk_idx.shape[0]
    M = T * TOP_K
    device = topk_idx.device

    flat_expert = topk_idx.reshape(-1).contiguous()
    flat_weight = topk_weights.reshape(-1).contiguous().to(torch.float32)

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(M, BLOCK_SIZE)

    # --- Pass 1: per-block partial histograms ---
    partial = torch.zeros((num_blocks, E_local), dtype=torch.int32, device=device)
    _count_kernel[(num_blocks,)](
        flat_expert, partial, M,
        local_expert_offset, E_local,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    # Reduce partials → token_counts, then build CSR offsets on device.
    token_counts = partial.sum(dim=0).to(torch.int32)                 # [E_local]
    expert_start = torch.empty(E_local + 1, dtype=torch.int32, device=device)
    expert_start[0] = 0
    torch.cumsum(token_counts, dim=0, out=expert_start[1:])
    N = int(expert_start[-1].item())  # one D→H sync — unavoidable for output sizing

    # --- Pass 2: atomic scatter ---
    sorted_token  = torch.empty(N, dtype=torch.int32, device=device)
    sorted_weight = torch.empty(N, dtype=flat_weight.dtype, device=device)

    # Cursor: mutable copy of the starts that atomic_add will advance.
    cursor = expert_start[:E_local].clone()

    _scatter_kernel[(num_blocks,)](
        flat_expert, flat_weight,
        cursor,
        sorted_token, sorted_weight,
        M,
        TOP_K=TOP_K,
        lo=local_expert_offset,
        E_local=E_local,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return sorted_token, sorted_weight, expert_start, token_counts

def build_tile_schedule(token_counts, BLOCK_M, E_local):
    device = token_counts.device

    # Per-expert tile counts and exclusive prefix sum.
    tiles_per_expert = (token_counts + BLOCK_M - 1) // BLOCK_M              # [E_local]
    expert_start_tile = torch.empty(E_local + 1, dtype=torch.int64, device=device)
    expert_start_tile[0] = 0
    torch.cumsum(tiles_per_expert, dim=0, out=expert_start_tile[1:])
    total_tiles = expert_start_tile[E_local]                                # 0-d tensor, stays on device

    total_tiles_int = int(total_tiles.item())

    tile_idx = torch.arange(total_tiles_int, dtype=torch.int64, device=device)

    # searchsorted on expert_start_tile[1:E_local+1] (the end-offsets):
    # bucketize each tile index into its expert.
    tile_expert_ids = torch.searchsorted(
        expert_start_tile[1:], tile_idx, right=True
    ).to(torch.int32)                                                       # [total_tiles]

    # tile_local_m via gather on the start-offsets.
    tile_local_m = (tile_idx - expert_start_tile[tile_expert_ids.long()]).to(torch.int32)

    return tile_expert_ids, tile_local_m, total_tiles

@triton.jit
def _topk_group_select(
    swb_g: tl.tensor,  # [GROUP_SIZE] s_with_bias for one group
    GROUP_SIZE: tl.constexpr,
):
    """Return sum of top-2 values in a single group. Helper for readability."""
    # Find max
    max1 = tl.max(swb_g, axis=0)
    # Mask first occurrence to find second max
    gs_offs = tl.arange(0, GROUP_SIZE)
    is_max1 = (swb_g == max1)
    # Zero out all matches and prior — get first match index
    first_idx = tl.min(tl.where(is_max1, gs_offs, GROUP_SIZE))
    mask1 = gs_offs == first_idx
    swb_masked = tl.where(mask1, float('-inf'), swb_g)
    max2 = tl.max(swb_masked, axis=0)
    return max1 + max2
 
 
@triton.jit
def moe_routing_kernel(
    # Inputs
    logits_ptr,        # [T, E_GLOBAL] float32
    bias_ptr,          # [E_GLOBAL] float32
    # Outputs
    topk_idx_ptr,      # [T, TOP_K] int32
    topk_weights_ptr,  # [T, TOP_K] float32
    # Strides
    logits_stride_t: tl.constexpr,
    # Scaling factor
    routed_scaling_factor,
    # Constants
    E_GLOBAL: tl.constexpr,    # 256
    N_GROUP: tl.constexpr,     # 8
    GROUP_SIZE: tl.constexpr,  # 32 (= E_GLOBAL / N_GROUP)
    TOPK_GROUP: tl.constexpr,  # 4
    TOP_K: tl.constexpr,       # 8
):
    """
    Fused DeepSeek-V3 no-aux routing.
 
    For each token:
      1. s = sigmoid(logits), s_with_bias = s + bias
      2. Group experts into N_GROUP groups of GROUP_SIZE
      3. Per group: score = sum of top-2 s_with_bias values
      4. Select TOPK_GROUP groups with highest scores
      5. Within selected groups, pick global TOP_K experts by s_with_bias
      6. Weights = s (no bias) at selected experts, normalized, scaled
    """
    token_id = tl.program_id(0)
 
    # -- Load full expert vector for this token --
    e_offs = tl.arange(0, E_GLOBAL)
    logits = tl.load(logits_ptr + token_id * logits_stride_t + e_offs).to(tl.float32)
    bias = tl.load(bias_ptr + e_offs).to(tl.float32)
 
    # -- Sigmoid + bias --
    s = tl.sigmoid(logits)           # [E_GLOBAL]
    s_with_bias = s + bias           # [E_GLOBAL]
 
    # -- Per-group top-2 scores --
    # Compute as [N_GROUP, GROUP_SIZE] using 2D indexing
    g_offs = tl.arange(0, N_GROUP)
    gs_offs = tl.arange(0, GROUP_SIZE)
    flat_idx = g_offs[:, None] * GROUP_SIZE + gs_offs[None, :]  # [8, 32]
 
    swb_2d = tl.load(logits_ptr + token_id * logits_stride_t + flat_idx).to(tl.float32)
    swb_2d = tl.sigmoid(swb_2d) + tl.load(bias_ptr + flat_idx).to(tl.float32)
 
    # Top-2 per group → group scores
    max1 = tl.max(swb_2d, axis=1)                    # [N_GROUP]
    # For second max: mask out the first max value's first occurrence
    is_max1 = (swb_2d == max1[:, None])
    # Find column index of first max per group
    first_idx = tl.min(tl.where(is_max1, gs_offs[None, :], GROUP_SIZE), axis=1)  # [N_GROUP]
    # Mask only first occurrence
    mask_first = (gs_offs[None, :] == first_idx[:, None])
    swb_second = tl.where(mask_first, float('-inf'), swb_2d)
    max2 = tl.max(swb_second, axis=1)                # [N_GROUP]
    group_scores = max1 + max2                        # [N_GROUP]
 
    # -- Select top-TOPK_GROUP groups --
    # 4 iterations of max-and-mask on 8 values
    gs_temp = group_scores
    group_selected = tl.zeros([N_GROUP], dtype=tl.int32)
 
    for _k in tl.static_range(TOPK_GROUP):
        best_val = tl.max(gs_temp, axis=0)
        is_best = (gs_temp == best_val) & (group_selected == 0)
        # Take first occurrence only
        best_g = tl.min(tl.where(is_best, g_offs, N_GROUP))
        group_selected = tl.where(g_offs == best_g, 1, group_selected)
        gs_temp = tl.where(group_selected != 0, float('-inf'), gs_temp)
 
    # -- Build flat mask over 256 experts: 1 if expert's group is selected --
    expert_group = e_offs // GROUP_SIZE  # [E_GLOBAL], values 0-7
 
    # Expand group_selected[expert_group] → flat_mask
    # Since group_selected is a [N_GROUP] register tensor, we broadcast
    # by iterating over groups (only 8 iterations, fully unrolled)
    flat_mask = tl.zeros([E_GLOBAL], dtype=tl.int32)
    for g in tl.static_range(N_GROUP):
        g_sel = tl.sum(tl.where(g_offs == g, group_selected, 0))
        flat_mask = tl.where((expert_group == g) & (g_sel != 0), 1, flat_mask)
 
    # -- Global top-8 within selected groups --
    swb_pruned = tl.where(flat_mask != 0, s_with_bias, float('-inf'))
 
    # 8 iterations of max-and-mask
    topk_indices = tl.zeros([TOP_K], dtype=tl.int32)
    topk_s_vals = tl.zeros([TOP_K], dtype=tl.float32)
    k_offs = tl.arange(0, TOP_K)
 
    for ki in tl.static_range(TOP_K):
        best = tl.max(swb_pruned, axis=0)
        is_best = (swb_pruned == best)
        best_idx = tl.min(tl.where(is_best, e_offs, E_GLOBAL))
 
        topk_indices = tl.where(k_offs == ki, best_idx, topk_indices)
 
        # Gather s[best_idx] (without bias) for weight computation
        s_at_best = tl.sum(tl.where(e_offs == best_idx, s, 0.0))
        topk_s_vals = tl.where(k_offs == ki, s_at_best, topk_s_vals)
 
        # Mask out selected expert
        swb_pruned = tl.where(e_offs == best_idx, float('-inf'), swb_pruned)
 
    # -- Weight normalization --
    weights_sum = tl.sum(topk_s_vals) + 1e-20
    topk_weights = (topk_s_vals / weights_sum) * routed_scaling_factor
 
    # -- Store --
    tl.store(topk_idx_ptr + token_id * TOP_K + k_offs, topk_indices)
    tl.store(topk_weights_ptr + token_id * TOP_K + k_offs, topk_weights)

def _check_cuda_and_move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device.type == 'cuda':
        return t
    if device.type != 'cuda':
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return t.to(device, non_blocking=True)


def _ensure_cuda(*tensors):
    # Ensure CUDA is available. If not, raise clear error.
    if not torch.cuda.is_available():
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                raise RuntimeError("CUDA inputs provided but CUDA is reported unavailable.")
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return torch.device('cuda')


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor
):
    # Constants per spec
    H = 7168
    I = 2048
    I2 = I * 2
    E_global = 256
    E_local = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    BLOCK = 128
    NUM_H_BLOCKS = H // BLOCK            # 56
    NUM_I_BLOCKS = I // BLOCK            # 16
    NUM_G1_BLOCKS = (2 * I) // BLOCK     # 32

    # Validate shapes and dtypes
    assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be FLOAT8_E4M3FN"
    assert gemm1_weights.dtype == torch.float8_e4m3fn, "gemm1_weights must be FLOAT8_E4M3FN"
    assert gemm2_weights.dtype == torch.float8_e4m3fn, "gemm2_weights must be FLOAT8_E4M3FN"
    assert routing_logits.dtype == torch.float32, "routing_logits must be float32"
    assert routing_bias.dtype in (torch.float32, torch.bfloat16, torch.float16), "routing_bias must be float or bf16/fp16"
    assert hidden_states_scale.dtype == torch.float32, "hidden_states_scale must be float32"
    assert gemm1_weights_scale.dtype == torch.float32, "gemm1_weights_scale must be float32"
    assert gemm2_weights_scale.dtype == torch.float32, "gemm2_weights_scale must be float32"

    T = int(routing_logits.shape[0])
    assert routing_logits.shape[-1] == E_global, "routing_logits last dim must be 256"
    assert hidden_states.shape == (T, H), "hidden_states must be [T, 7168]"
    assert hidden_states_scale.shape == (NUM_H_BLOCKS, T), "hidden_states_scale must be [56, T]"
    assert gemm1_weights.shape == (E_local, 2 * I, H), "gemm1_weights must be [32, 4096, 7168]"
    assert gemm1_weights_scale.shape == (E_local, NUM_G1_BLOCKS, NUM_H_BLOCKS), "gemm1_weights_scale must be [32, 32, 56]"
    assert gemm2_weights.shape == (E_local, H, I), "gemm2_weights must be [32, 7168, 2048]"
    assert gemm2_weights_scale.shape == (E_local, NUM_H_BLOCKS, NUM_I_BLOCKS), "gemm2_weights_scale must be [32, 56, 16]"

    # Device management
    device = _ensure_cuda(routing_logits, routing_bias, hidden_states, hidden_states_scale,
                          gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale)
    orig_device = routing_logits.device

    # Move tensors to CUDA if needed
    routing_logits_cu = _check_cuda_and_move(routing_logits, device).contiguous()
    routing_bias_cu = _check_cuda_and_move(routing_bias.to(torch.float32), device).contiguous()
    hidden_states_cu = _check_cuda_and_move(hidden_states, device).contiguous()
    hidden_states_scale_cu = _check_cuda_and_move(hidden_states_scale, device).contiguous()
    gemm1_weights_cu = _check_cuda_and_move(gemm1_weights, device).contiguous()
    gemm1_weights_scale_cu = _check_cuda_and_move(gemm1_weights_scale, device).contiguous()
    gemm2_weights_cu = _check_cuda_and_move(gemm2_weights, device).contiguous()
    gemm2_weights_scale_cu = _check_cuda_and_move(gemm2_weights_scale, device).contiguous()

    # 1) Routing (DeepSeek-V3 no-aux) on CUDA (PyTorch)
    logits = routing_logits_cu.to(torch.float32)                      # [T, E]
    bias = routing_bias_cu.to(torch.float32).reshape(-1)              # [E]

    # Allocate outputs for routing kernel
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    # Launch fused routing kernel — one program per token
    grid = (T,)
    moe_routing_kernel[grid](
        logits, bias,
        topk_idx, topk_weights,
        logits.stride(0),
        routed_scaling_factor,
        E_global, N_GROUP, E_local, TOPK_GROUP, TOP_K,
    )

    # GEMM 1 Constants
    BLOCK_M_1 = 64
    BLOCK_N_1 = 64 # BLOCK_I 
    BLOCK_K_1 = 128 
    GROUP_SIZE_M_1 = 4
    warps_1 = 4
    stages_1 = 3

    sorted_token_ids, sorted_weights, expert_starts, token_counts = build_expert_dispatch(
        topk_idx,
        topk_weights,
        local_expert_offset,
        E_local,
        TOP_K
    )
    tile_expert_ids, tile_local_m, total_tiles = build_tile_schedule(
        token_counts,
        BLOCK_M_1,
        E_local
    )
    total_tiles = total_tiles.item() 

    # Pack A to coalesce
    A_packed = torch.index_select(hidden_states_cu, dim=0, index=sorted_token_ids).contiguous()
    A_scale_packed = torch.index_select(hidden_states_scale_cu, dim=1, index=sorted_token_ids).contiguous()

    # GEMM 1 accumulators
    total_selected = sorted_token_ids.shape[0]
    gemm1_out = torch.zeros((total_selected, I), dtype=torch.float8_e4m3fn, device=device)
    gemm1_scale_out = torch.zeros((triton.cdiv(I, BLOCK_N_1), total_selected), dtype=torch.float32, device=device)

    # Launch GEMM 1
    gemm1_grid = (total_tiles * triton.cdiv(I, BLOCK_N_1),) # cdiv not technically necessary
    moe_gemm1_kernel[gemm1_grid](
        A_packed, A_scale_packed,
        A_packed.stride(0), A_packed.stride(1),
        A_scale_packed.stride(0), A_scale_packed.stride(1),
        gemm1_weights_cu, gemm1_weights_scale_cu,
        gemm1_weights_cu.stride(0), gemm1_weights_cu.stride(1), gemm1_weights_cu.stride(2),
        gemm1_weights_scale_cu.stride(0), gemm1_weights_scale_cu.stride(1), gemm1_weights_scale_cu.stride(2),
        gemm1_out, gemm1_scale_out,
        gemm1_out.stride(0), gemm1_out.stride(1),
        gemm1_scale_out.stride(0), gemm1_scale_out.stride(1),
        sorted_token_ids,
        expert_starts,
        tile_expert_ids,
        tile_local_m,
        T, H, I, I2,
        BLOCK_M_1, BLOCK_N_1, BLOCK_K_1,
        GROUP_SIZE_M_1, total_tiles,
        num_warps=warps_1,
        num_stages=stages_1
    )

    # GEMM 2 Constants
    BLOCK_M_2 = 64
    BLOCK_N_2 = 64  
    BLOCK_K_2 = 64 
    GROUP_SIZE_M_2 = 4
    warps_2 = 4
    stages_2 = 4

    # GEMM 2 accumulator
    gemm2_out = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Launch GEMM 2
    gemm2_grid = (total_tiles * triton.cdiv(H, BLOCK_N_2),) # cdiv not technically necessary
    moe_gemm2_kernel[gemm2_grid](
        gemm1_out, gemm1_scale_out,
        gemm1_out.stride(0), gemm1_out.stride(1),
        gemm1_scale_out.stride(0), gemm1_scale_out.stride(1),
        gemm2_weights_cu, gemm2_weights_scale_cu,
        gemm2_weights_cu.stride(0), gemm2_weights_cu.stride(1), gemm2_weights_cu.stride(2),
        gemm2_weights_scale_cu.stride(0), gemm2_weights_scale_cu.stride(1), gemm2_weights_scale_cu.stride(2),
        sorted_weights, 
        gemm2_out,
        gemm2_out.stride(0), gemm2_out.stride(1),
        sorted_token_ids, 
        expert_starts,
        tile_expert_ids,
        tile_local_m,
        T, H, I,
        BLOCK_M_2, BLOCK_N_2, BLOCK_K_2, BLOCK_N_1,
        GROUP_SIZE_M_2, total_tiles,
        num_warps=warps_2,
        num_stages=stages_2
    )

    output.copy_(gemm2_out)
"""Triton GPU kernels for alpha factor operations.

All kernels operate on float32 torch tensors on CUDA.
Requires: triton >= 2.1, torch with CUDA support.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused rolling mean + std (Welford online algorithm)
# ---------------------------------------------------------------------------

@triton.jit
def _rolling_mean_std_kernel(
    input_ptr,
    mean_out_ptr,
    std_out_ptr,
    stride_t,
    stride_s,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Compute rolling mean and std over time axis for 2D (T, S) tensor.

    Each program handles one output time-step and a block of symbols.
    Uses Welford's online algorithm for numerical stability.
    """
    pid_t = tl.program_id(0)  # output row index (0 .. T-W)
    pid_s = tl.program_id(1)  # symbol block index

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Welford accumulators
    count = tl.zeros([BLOCK_S], dtype=tl.float32)
    mean = tl.zeros([BLOCK_S], dtype=tl.float32)
    m2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    for w in range(W):
        t_idx = pid_t + w
        ptrs = input_ptr + t_idx * stride_t + s_offsets * stride_s
        vals = tl.load(ptrs, mask=s_mask, other=float("nan"))
        valid = vals == vals  # NaN check: NaN != NaN

        new_count = count + tl.where(valid, 1.0, 0.0)
        safe_count = tl.where(new_count > 0, new_count, 1.0)
        delta = tl.where(valid, vals - mean, 0.0)
        new_mean = mean + delta / safe_count
        delta2 = tl.where(valid, vals - new_mean, 0.0)
        new_m2 = m2 + delta * delta2

        count = new_count
        mean = tl.where(new_count > count - 1, new_mean, mean)
        m2 = new_m2

    # Output row = pid_t + W - 1
    out_t = pid_t + W - 1
    out_ptrs_mean = mean_out_ptr + out_t * stride_t + s_offsets * stride_s
    out_ptrs_std = std_out_ptr + out_t * stride_t + s_offsets * stride_s

    safe_count = tl.where(count > 0, count, 1.0)
    variance = m2 / safe_count
    std = tl.sqrt(variance)

    # Write NaN where count == 0
    final_mean = tl.where(count > 0, mean, float("nan"))
    final_std = tl.where(count > 0, std, float("nan"))

    tl.store(out_ptrs_mean, final_mean, mask=s_mask)
    tl.store(out_ptrs_std, final_std, mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel 2: Rolling reduce (sum / max / min)
# ---------------------------------------------------------------------------

@triton.jit
def _rolling_reduce_kernel(
    input_ptr,
    output_ptr,
    stride_t,
    stride_s,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MODE: tl.constexpr,  # 0=sum, 1=max, 2=min
):
    """Rolling reduce over time axis. MODE: 0=nansum, 1=nanmax, 2=nanmin."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    if MODE == 0:
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    elif MODE == 1:
        acc = tl.full([BLOCK_S], value=float("-inf"), dtype=tl.float32)
    else:
        acc = tl.full([BLOCK_S], value=float("inf"), dtype=tl.float32)

    count = tl.zeros([BLOCK_S], dtype=tl.float32)

    for w in range(W):
        t_idx = pid_t + w
        ptrs = input_ptr + t_idx * stride_t + s_offsets * stride_s
        vals = tl.load(ptrs, mask=s_mask, other=float("nan"))
        valid = vals == vals

        count += tl.where(valid, 1.0, 0.0)
        if MODE == 0:
            acc += tl.where(valid, vals, 0.0)
        elif MODE == 1:
            acc = tl.where(valid & (vals > acc), vals, acc)
        else:
            acc = tl.where(valid & (vals < acc), vals, acc)

    out_t = pid_t + W - 1
    out_ptrs = output_ptr + out_t * stride_t + s_offsets * stride_s
    result = tl.where(count > 0, acc, float("nan"))
    tl.store(out_ptrs, result, mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel 3: Parallel EMA via prefix scan
# ---------------------------------------------------------------------------

@triton.jit
def _parallel_ema_scan_kernel(
    input_ptr,
    output_ptr,
    stride_t,
    stride_s,
    alpha_val,
    T: tl.constexpr,
    S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """EMA scan over time axis, parallelized across BLOCK_S symbol columns.

    EMA recurrence: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    Each symbol column is independent — we process BLOCK_S columns in
    parallel on one SM while scanning sequentially over time.
    This avoids the Python-loop overhead of the torch backend.
    """
    pid_s = tl.program_id(0)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    one_minus_alpha = 1.0 - alpha_val

    # Sequential scan over T; each of BLOCK_S columns is independent.
    prev_val = tl.zeros([BLOCK_S], dtype=tl.float32)
    prev_valid = tl.full([BLOCK_S], value=0, dtype=tl.int32)

    for t in range(T):
        cur_x_ptrs = input_ptr + t * stride_t + s_offsets * stride_s
        cur_x = tl.load(cur_x_ptrs, mask=s_mask, other=float("nan"))
        cur_valid = (cur_x == cur_x)

        # EMA update with NaN handling:
        # If current NaN: keep previous
        # If previous invalid: use current
        # Else: alpha * cur + (1-alpha) * prev
        new_val = tl.where(
            cur_valid,
            tl.where(
                prev_valid > 0,
                alpha_val * cur_x + one_minus_alpha * prev_val,
                cur_x,
            ),
            prev_val,
        )
        new_valid_flag = tl.where(cur_valid | (prev_valid > 0), 1, 0)

        out_ptrs = output_ptr + t * stride_t + s_offsets * stride_s
        out_val = tl.where(new_valid_flag > 0, new_val, float("nan"))
        tl.store(out_ptrs, out_val, mask=s_mask)

        prev_val = new_val
        prev_valid = new_valid_flag


# ---------------------------------------------------------------------------
# Kernel 4: Fused rolling correlation / covariance
# ---------------------------------------------------------------------------

@triton.jit
def _rolling_corr_cov_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    stride_t,
    stride_s,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    IS_CORR: tl.constexpr,  # 1 = correlation, 0 = covariance
):
    """Fused rolling correlation or covariance in a single pass."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Accumulators for online stats
    count = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_x = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_y = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_xx = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_yy = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_xy = tl.zeros([BLOCK_S], dtype=tl.float32)

    for w in range(W):
        t_idx = pid_t + w
        x_ptrs = x_ptr + t_idx * stride_t + s_offsets * stride_s
        y_ptrs = y_ptr + t_idx * stride_t + s_offsets * stride_s
        xv = tl.load(x_ptrs, mask=s_mask, other=float("nan"))
        yv = tl.load(y_ptrs, mask=s_mask, other=float("nan"))
        valid = (xv == xv) & (yv == yv)

        safe_x = tl.where(valid, xv, 0.0)
        safe_y = tl.where(valid, yv, 0.0)
        count += tl.where(valid, 1.0, 0.0)
        sum_x += safe_x
        sum_y += safe_y
        sum_xx += safe_x * safe_x
        sum_yy += safe_y * safe_y
        sum_xy += safe_x * safe_y

    safe_count = tl.where(count > 0, count, 1.0)
    mean_x = sum_x / safe_count
    mean_y = sum_y / safe_count
    cov = sum_xy / safe_count - mean_x * mean_y

    if IS_CORR:
        var_x = sum_xx / safe_count - mean_x * mean_x
        var_y = sum_yy / safe_count - mean_y * mean_y
        denom = tl.sqrt(tl.maximum(var_x, 0.0) * tl.maximum(var_y, 0.0)) + 1e-12
        result = tl.minimum(tl.maximum(cov / denom, -1.0), 1.0)
    else:
        result = cov

    result = tl.where(count > 0, result, float("nan"))

    out_t = pid_t + W - 1
    out_ptrs = output_ptr + out_t * stride_t + s_offsets * stride_s
    tl.store(out_ptrs, result, mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel 5: Cross-sectional rank (per-row ranking)
# ---------------------------------------------------------------------------

@triton.jit
def _cs_rank_kernel(
    input_ptr,
    output_ptr,
    stride_t,
    stride_s,
    T: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Cross-sectional rank per row, normalized to [1/N, 1].

    For each row, count how many valid values are less than each element,
    then normalize by total valid count.
    """
    pid_t = tl.program_id(0)

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Load the full row
    ptrs = input_ptr + pid_t * stride_t + s_offsets * stride_s
    vals = tl.load(ptrs, mask=s_mask, other=float("nan"))
    valid = vals == vals
    n_valid = tl.sum(tl.where(valid, 1.0, 0.0))

    # For each element, count how many valid values are strictly less
    # This is O(S^2) per row but S <= 256, so it's fine in a single block
    # Use broadcasting: compare vals[i] against all vals[j]
    # Triton doesn't support 2D within a kernel easily, so we loop

    ranks = tl.zeros([BLOCK_S], dtype=tl.float32)
    for j in range(BLOCK_S):
        if j < S:
            other_val = tl.load(input_ptr + pid_t * stride_t + j * stride_s)
            other_valid = other_val == other_val
            # Count: other_val < vals[i] and both valid
            less = tl.where(
                valid & other_valid & (other_val < vals),
                1.0,
                0.0,
            )
            # Tie-breaking: if equal, count by index (j < i)
            equal_lower_idx = tl.where(
                valid & other_valid & (other_val == vals) & (j < s_offsets),
                1.0,
                0.0,
            )
            ranks += less + equal_lower_idx

    # Normalize: rank = (position + 1) / n_valid
    safe_n = tl.where(n_valid > 0, n_valid, 1.0)
    normalized = (ranks + 1.0) / safe_n
    result = tl.where(valid, normalized, float("nan"))

    out_ptrs = output_ptr + pid_t * stride_t + s_offsets * stride_s
    tl.store(out_ptrs, result, mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel 6: Decay linear (linearly weighted rolling sum)
# ---------------------------------------------------------------------------

@triton.jit
def _decay_linear_kernel(
    input_ptr,
    output_ptr,
    stride_t,
    stride_s,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Linearly weighted rolling average: weights = [1, 2, ..., W]."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    weighted_sum = tl.zeros([BLOCK_S], dtype=tl.float32)
    weight_sum = tl.zeros([BLOCK_S], dtype=tl.float32)

    for w in range(W):
        t_idx = pid_t + w
        ptrs = input_ptr + t_idx * stride_t + s_offsets * stride_s
        vals = tl.load(ptrs, mask=s_mask, other=float("nan"))
        valid = vals == vals

        weight = tl.full([BLOCK_S], value=(w + 1), dtype=tl.float32)
        weighted_sum += tl.where(valid, vals * weight, 0.0)
        weight_sum += tl.where(valid, weight, 0.0)

    safe_denom = tl.where(weight_sum > 0, weight_sum, 1.0)
    result = weighted_sum / safe_denom
    result = tl.where(weight_sum > 0, result, float("nan"))

    out_t = pid_t + W - 1
    out_ptrs = output_ptr + out_t * stride_t + s_offsets * stride_s
    tl.store(out_ptrs, result, mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel 7: Batch rank IC (Pearson correlation per row, batched over factors)
# ---------------------------------------------------------------------------

@triton.jit
def _batch_rank_ic_kernel(
    alpha_ptr,
    returns_ptr,
    ic_row_out_ptr,
    valid_count_out_ptr,
    stride_n,
    stride_t,
    stride_s,
    ret_stride_t,
    ret_stride_s,
    N: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Compute per-row Pearson correlation for batch of factors.

    alpha: (N, T, S), returns: (T, S)
    Output: ic_row_out (N, T) — per-row IC, valid_count_out (N, T) — valid flag.
    """
    pid_n = tl.program_id(0)  # factor index
    pid_t = tl.program_id(1)  # time step

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    # Load alpha row and returns row
    a_ptrs = alpha_ptr + pid_n * stride_n + pid_t * stride_t + s_offsets * stride_s
    r_ptrs = returns_ptr + pid_t * ret_stride_t + s_offsets * ret_stride_s

    a_vals = tl.load(a_ptrs, mask=s_mask, other=float("nan"))
    r_vals = tl.load(r_ptrs, mask=s_mask, other=float("nan"))

    valid = (a_vals == a_vals) & (r_vals == r_vals)
    count = tl.sum(tl.where(valid, 1.0, 0.0))

    safe_a = tl.where(valid, a_vals, 0.0)
    safe_r = tl.where(valid, r_vals, 0.0)
    safe_count = tl.where(count > 0, count, 1.0)

    mean_a = tl.sum(safe_a) / safe_count
    mean_r = tl.sum(safe_r) / safe_count

    ca = tl.where(valid, a_vals - mean_a, 0.0)
    cr = tl.where(valid, r_vals - mean_r, 0.0)

    cov = tl.sum(ca * cr)
    var_a = tl.sum(ca * ca)
    var_r = tl.sum(cr * cr)

    denom = tl.sqrt(var_a * var_r) + 1e-12
    ic = cov / denom
    ic = tl.where((count >= 2) & (var_a > 1e-24) & (var_r > 1e-24), ic, 0.0)

    # Store per-row IC
    out_ptr = ic_row_out_ptr + pid_n * T + pid_t
    tl.store(out_ptr, ic)

    count_ptr = valid_count_out_ptr + pid_n * T + pid_t
    tl.store(count_ptr, count)


# ---------------------------------------------------------------------------
# Kernel 8: Pairwise factor correlation matrix
# ---------------------------------------------------------------------------

@triton.jit
def _factor_corr_matrix_kernel(
    factors_ptr,
    corr_out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute |Pearson correlation| between factor i and factor j.

    factors: (N, D) flattened signals. Output: corr_out (N, N).
    Only computes upper triangle; mirrors to lower.
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # Only compute upper triangle (including diagonal)
    if pid_j < pid_i:
        return

    # Accumulate dot products in blocks — use scalar accumulators
    acc_sum_x: tl.float32 = 0.0
    acc_sum_y: tl.float32 = 0.0
    acc_sum_xx: tl.float32 = 0.0
    acc_sum_yy: tl.float32 = 0.0
    acc_sum_xy: tl.float32 = 0.0
    acc_count: tl.float32 = 0.0

    for block_start in range(0, D, BLOCK_D):
        d_offsets = block_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        x_ptrs = factors_ptr + pid_i * D + d_offsets
        y_ptrs = factors_ptr + pid_j * D + d_offsets

        xv = tl.load(x_ptrs, mask=d_mask, other=float("nan"))
        yv = tl.load(y_ptrs, mask=d_mask, other=float("nan"))

        valid = (xv == xv) & (yv == yv)
        sx = tl.where(valid, xv, 0.0)
        sy = tl.where(valid, yv, 0.0)

        acc_count += tl.sum(tl.where(valid, 1.0, 0.0))
        acc_sum_x += tl.sum(sx)
        acc_sum_y += tl.sum(sy)
        acc_sum_xx += tl.sum(sx * sx)
        acc_sum_yy += tl.sum(sy * sy)
        acc_sum_xy += tl.sum(sx * sy)

    safe_count = tl.where(acc_count > 0, acc_count, 1.0)
    mean_x = acc_sum_x / safe_count
    mean_y = acc_sum_y / safe_count
    cov = acc_sum_xy / safe_count - mean_x * mean_y
    var_x = acc_sum_xx / safe_count - mean_x * mean_x
    var_y = acc_sum_yy / safe_count - mean_y * mean_y
    denom = tl.sqrt(tl.maximum(var_x, 0.0) * tl.maximum(var_y, 0.0)) + 1e-12
    corr = tl.abs(cov / denom)
    corr = tl.where(acc_count >= 5, corr, 0.0)

    # Store upper triangle
    tl.store(corr_out_ptr + pid_i * N + pid_j, corr)
    # Mirror to lower triangle
    if pid_j > pid_i:
        tl.store(corr_out_ptr + pid_j * N + pid_i, corr)

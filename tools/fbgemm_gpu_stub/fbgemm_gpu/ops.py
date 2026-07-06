"""
Register torch.ops.fbgemm.* operators with pure-PyTorch CPU/MPS implementations.

Only covers the ops used by torchrec's sparse/ and modules/ (non-distributed).
Anything not implemented raises NotImplementedError on first call.
"""

import torch

lib = torch.library.Library("fbgemm", "DEF")
impl_lib = torch.library.Library("fbgemm", "IMPL")


# ─── cumsum variants ───────────────────────────────────────────────────────────

lib.define("asynchronous_complete_cumsum(Tensor t) -> Tensor")


@torch.library.impl(impl_lib, "asynchronous_complete_cumsum", "CPU")
def _complete_cumsum_cpu(t: torch.Tensor) -> torch.Tensor:
    # complete cumsum: prepends a 0, output length = input length + 1
    return torch.cat([torch.zeros(1, dtype=t.dtype, device=t.device), t.cumsum(0)])


lib.define("asynchronous_inclusive_cumsum(Tensor t) -> Tensor")


@torch.library.impl(impl_lib, "asynchronous_inclusive_cumsum", "CPU")
def _inclusive_cumsum_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.cumsum(0)


lib.define("asynchronous_exclusive_cumsum(Tensor t) -> Tensor")


@torch.library.impl(impl_lib, "asynchronous_exclusive_cumsum", "CPU")
def _exclusive_cumsum_cpu(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, dtype=t.dtype, device=t.device), t.cumsum(0)[:-1]])


# ─── segment_sum_csr ──────────────────────────────────────────────────────────

lib.define("segment_sum_csr(int batch_size, Tensor offsets, Tensor values) -> Tensor")


@torch.library.impl(impl_lib, "segment_sum_csr", "CPU")
def _segment_sum_csr_cpu(
    batch_size: int, offsets: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    num_segments = offsets.numel() - 1
    out = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    for i in range(num_segments):
        start, end = offsets[i].item(), offsets[i + 1].item()
        out[i] = values[start:end].sum()
    return out


# ─── offsets_range ─────────────────────────────────────────────────────────────

lib.define("offsets_range(Tensor offsets, int range_size) -> Tensor")


@torch.library.impl(impl_lib, "offsets_range", "CPU")
def _offsets_range_cpu(offsets: torch.Tensor, range_size: int) -> torch.Tensor:
    # For each segment [offsets[i], offsets[i+1]), produce [0, 1, ..., length-1]
    out = torch.zeros(range_size, dtype=torch.long, device=offsets.device)
    for i in range(offsets.numel() - 1):
        start, end = offsets[i].item(), offsets[i + 1].item()
        out[start:end] = torch.arange(end - start, device=offsets.device)
    return out


# ─── jagged_to_padded_dense ───────────────────────────────────────────────────

lib.define(
    "jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value=0.0) -> Tensor"
)


@torch.library.impl(impl_lib, "jagged_to_padded_dense", "CPU")
def _jagged_to_padded_dense_cpu(
    values: torch.Tensor,
    offsets: list[torch.Tensor],
    max_lengths: list[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    # Single-level jagged -> padded dense
    offs = offsets[0]
    max_len = max_lengths[0]
    batch_size = offs.numel() - 1
    if values.dim() == 1:
        out = torch.full(
            (batch_size, max_len), padding_value, dtype=values.dtype, device=values.device
        )
    else:
        out = torch.full(
            (batch_size, max_len, *values.shape[1:]),
            padding_value,
            dtype=values.dtype,
            device=values.device,
        )
    for i in range(batch_size):
        start, end = offs[i].item(), offs[i + 1].item()
        length = min(end - start, max_len)
        out[i, :length] = values[start : start + length]
    return out


# ─── permute_1D_sparse_data ───────────────────────────────────────────────────

lib.define(
    "permute_1D_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights, int? permuted_lengths_sum) -> (Tensor, Tensor, Tensor?)"
)


@torch.library.impl(impl_lib, "permute_1D_sparse_data", "CPU")
def _permute_1d_sparse_data_cpu(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor = None,
    permuted_lengths_sum: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    offsets = torch.cat(
        [torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths.cumsum(0)]
    )
    permuted_lengths = lengths[permute]
    permuted_values_list = []
    permuted_weights_list = []
    for idx in permute.tolist():
        start, end = offsets[idx].item(), offsets[idx + 1].item()
        permuted_values_list.append(values[start:end])
        if weights is not None:
            permuted_weights_list.append(weights[start:end])
    permuted_values = (
        torch.cat(permuted_values_list) if permuted_values_list else values.new_empty(0)
    )
    permuted_weights = torch.cat(permuted_weights_list) if weights is not None else None
    return permuted_lengths, permuted_values, permuted_weights


# ─── permute_2D_sparse_data ───────────────────────────────────────────────────

lib.define(
    "permute_2D_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights, int? permuted_lengths_sum) -> (Tensor, Tensor, Tensor?)"
)


@torch.library.impl(impl_lib, "permute_2D_sparse_data", "CPU")
def _permute_2d_sparse_data_cpu(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor = None,
    permuted_lengths_sum: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # lengths shape: (T, B) — permute reorders the T dimension
    T, B = lengths.shape
    permuted_lengths = lengths[permute]
    offsets = torch.zeros(T * B + 1, dtype=lengths.dtype, device=lengths.device)
    offsets[1:] = lengths.reshape(-1).cumsum(0)
    permuted_values_list = []
    permuted_weights_list = []
    for new_t, old_t in enumerate(permute.tolist()):
        for b in range(B):
            flat_idx = old_t * B + b
            start, end = offsets[flat_idx].item(), offsets[flat_idx + 1].item()
            permuted_values_list.append(values[start:end])
            if weights is not None:
                permuted_weights_list.append(weights[start:end])
    permuted_values = (
        torch.cat(permuted_values_list) if permuted_values_list else values.new_empty(0)
    )
    permuted_weights = torch.cat(permuted_weights_list) if weights is not None else None
    return permuted_lengths, permuted_values, permuted_weights


lib.define(
    "permute_2D_sparse_data_input1D(Tensor permute, Tensor lengths, Tensor values, Tensor? weights, int? permuted_lengths_sum, int stride) -> (Tensor, Tensor, Tensor?)"
)


@torch.library.impl(impl_lib, "permute_2D_sparse_data_input1D", "CPU")
def _permute_2d_sparse_data_input1d_cpu(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor = None,
    permuted_lengths_sum: int = None,
    stride: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # Same as 2D but lengths is flattened (1D) with known stride (B)
    B = stride
    T = lengths.numel() // B
    lengths_2d = lengths.reshape(T, B)
    return _permute_2d_sparse_data_cpu(permute, lengths_2d, values, weights, permuted_lengths_sum)


# ─── group_index_select_dim0 ──────────────────────────────────────────────────

lib.define("group_index_select_dim0(Tensor[] inputs, Tensor[] indices) -> Tensor[]")


@torch.library.impl(impl_lib, "group_index_select_dim0", "CPU")
def _group_index_select_dim0_cpu(
    inputs: list[torch.Tensor],
    indices: list[torch.Tensor],
) -> list[torch.Tensor]:
    return [inp[idx] for inp, idx in zip(inputs, indices)]


# ─── permute_pooled_embs_auto_grad ────────────────────────────────────────────

lib.define(
    "permute_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor"
)


@torch.library.impl(impl_lib, "permute_pooled_embs_auto_grad", "CPU")
def _permute_pooled_embs_cpu(
    pooled_embs: torch.Tensor,
    offset_dim_list: torch.Tensor,
    permute_list: torch.Tensor,
    inv_offset_dim_list: torch.Tensor,
    inv_permute_list: torch.Tensor,
) -> torch.Tensor:
    # Reorder columns of pooled_embs according to permute_list
    # offset_dim_list[i] = start column of feature i, offset_dim_list[i+1] = end
    num_features = permute_list.numel()
    slices = []
    for i in range(num_features):
        src = permute_list[i].item()
        start = offset_dim_list[src].item()
        end = offset_dim_list[src + 1].item()
        slices.append(pooled_embs[:, start:end])
    return torch.cat(slices, dim=1)


# ─── keyed_jagged_index_select_dim1 ───────────────────────────────────────────

lib.define(
    "keyed_jagged_index_select_dim1(Tensor values, Tensor lengths, Tensor offsets, Tensor indices, int batch_size, Tensor? weights, int? selected_lengths_sum) -> (Tensor, Tensor, Tensor?)"
)


@torch.library.impl(impl_lib, "keyed_jagged_index_select_dim1", "CPU")
def _keyed_jagged_index_select_dim1_cpu(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    weights: torch.Tensor = None,
    selected_lengths_sum: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # Index-select along the batch dimension for each key
    num_keys = lengths.numel() // batch_size
    selected_lengths_list = []
    selected_values_list = []
    selected_weights_list = []
    flat_offset = 0
    for k in range(num_keys):
        key_lengths = lengths[k * batch_size : (k + 1) * batch_size]
        key_offsets = torch.cat(
            [
                torch.zeros(1, dtype=key_lengths.dtype, device=key_lengths.device),
                key_lengths.cumsum(0),
            ]
        )
        for idx in indices.tolist():
            start = (flat_offset + key_offsets[idx]).long().item()
            end = (flat_offset + key_offsets[idx + 1]).long().item()
            selected_values_list.append(values[start:end])
            selected_lengths_list.append(end - start)
            if weights is not None:
                selected_weights_list.append(weights[start:end])
        flat_offset += key_offsets[-1].item()
    selected_lengths = torch.tensor(
        selected_lengths_list, dtype=lengths.dtype, device=lengths.device
    )
    selected_values = (
        torch.cat(selected_values_list) if selected_values_list else values.new_empty(0)
    )
    selected_weights = torch.cat(selected_weights_list) if weights is not None else None
    return selected_values, selected_lengths, selected_weights


# ─── permute_multi_embedding ──────────────────────────────────────────────────

lib.define(
    "permute_multi_embedding(Tensor pooled_embs, Tensor permutes, Tensor in_shape, Tensor out_shape, Tensor out_lengths) -> Tensor"
)


@torch.library.impl(impl_lib, "permute_multi_embedding", "CPU")
def _permute_multi_embedding_cpu(
    pooled_embs: torch.Tensor,
    permutes: torch.Tensor,
    in_shape: torch.Tensor,
    out_shape: torch.Tensor,
    out_lengths: torch.Tensor,
) -> torch.Tensor:
    # permutes is (N, 4): [in_tensor, out_tensor, in_offset, out_offset, length, ...]
    # Simple column-reorder of a concatenated embedding tensor
    batch_size = pooled_embs.shape[0]
    total_out = out_lengths.sum().item()
    out = torch.empty(batch_size, total_out, dtype=pooled_embs.dtype, device=pooled_embs.device)
    for row in permutes:
        _in_tensor, _out_tensor, in_offset, out_offset = row[:4].tolist()
        length = (
            row[4].item()
            if row.numel() > 4
            else (out_lengths[_out_tensor].item() if _out_tensor < out_lengths.numel() else 0)
        )
        out[:, out_offset : out_offset + length] = pooled_embs[:, in_offset : in_offset + length]
    return out


# ─── new_unified_tensor ───────────────────────────────────────────────────────

lib.define("new_unified_tensor(Tensor existing, int[] sizes, bool pinned) -> Tensor")


@torch.library.impl(impl_lib, "new_unified_tensor", "CPU")
def _new_unified_tensor_cpu(
    existing: torch.Tensor,
    sizes: list[int],
    pinned: bool,
) -> torch.Tensor:
    return torch.empty(sizes, dtype=existing.dtype, device=existing.device)


# ─── jagged_index_select_2d_forward_v2 ────────────────────────────────────────

lib.define(
    "jagged_index_select_2d_forward_v2(Tensor values, Tensor indices, Tensor input_offsets, Tensor output_offsets, int? num_dense_output_rows) -> Tensor"
)


@torch.library.impl(impl_lib, "jagged_index_select_2d_forward_v2", "CPU")
def _jagged_index_select_2d_fwd_cpu(
    values: torch.Tensor,
    indices: torch.Tensor,
    input_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    num_dense_output_rows: int = None,
) -> torch.Tensor:
    results = []
    for idx in indices.tolist():
        start = input_offsets[idx].item()
        end = input_offsets[idx + 1].item()
        results.append(values[start:end])
    return torch.cat(results) if results else values.new_empty(0, *values.shape[1:])


# ─── Unimplemented stubs (raise on use) ──────────────────────────────────────


def _not_implemented(name: str):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"torch.ops.fbgemm.{name} is not implemented in the CPU/MPS stub. "
            "This op is only needed for advanced features (ZCH, ITEP pruning, regrouping)."
        )

    return _fn


_UNIMPLEMENTED = [
    (
        "kt_regroup_arguments(Tensor keys, Tensor lengths, Tensor values) -> (Tensor, Tensor, Tensor, Tensor)",
        "kt_regroup_arguments",
    ),
    (
        "regroup_keyed_tensor(Tensor keys, Tensor lengths, Tensor values, Tensor permutes, Tensor in_shape, Tensor out_shape, Tensor out_lengths) -> Tensor",
        "regroup_keyed_tensor",
    ),
    ("create_zch_buffer(int size, int min_len, int max_len) -> Tensor", "create_zch_buffer"),
    ("init_address_lookup(Tensor address, int idx) -> ()", "init_address_lookup"),
    (
        "zero_collision_hash(Tensor input, Tensor identities, Tensor eviction_policy, int idx) -> (Tensor, Tensor)",
        "zero_collision_hash",
    ),
    (
        "prune_embedding_tables(Tensor indices, Tensor lengths, Tensor offsets, Tensor hash_sizes) -> (Tensor, Tensor, Tensor)",
        "prune_embedding_tables",
    ),
    (
        "remap_indices_update_utils(Tensor indices, Tensor mapping, int hash_size) -> Tensor",
        "remap_indices_update_utils",
    ),
]

for _schema, _name in _UNIMPLEMENTED:
    try:
        lib.define(_schema)
        impl_lib.impl(_name, _not_implemented(_name), "CPU")
    except Exception:
        pass

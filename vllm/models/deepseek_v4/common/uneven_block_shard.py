# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uneven block-aligned tensor sharding for the DeepSeek-V4 shared expert.

The shared expert is block-FP8 quantized (``weight_block_size=[128, 128]``). Under
tensor parallelism its intermediate dimension is sharded across ranks, but a block
scale is shared by a whole 128-block, so a rank's shard must be a whole number of
blocks. When ``tp_size`` does not divide ``intermediate_size / block`` (e.g.
``moe_intermediate_size=3072`` with block 128 at TP=16: ``3072 / 16 = 192`` is not a
multiple of 128) the usual even split is impossible.

This module distributes the ``intermediate / block`` whole blocks across ranks as
evenly as possible (the first ``n_blocks % tp_size`` ranks get one extra block). Every
rank then owns a whole number of blocks, so FP8 block scales slice cleanly. The
per-rank ``down_proj`` partials sum to the full output exactly as in the even-TP case,
so the reduction (``reduce_results`` all-reduce, or the FusedMoE combine) is unchanged.
"""

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


def compute_block_shards(n_blocks: int, tp_size: int) -> list[tuple[int, int]]:
    """Return ``(block_start, n_local_blocks)`` for each rank.

    Distributes ``n_blocks`` whole blocks across ``tp_size`` ranks as evenly as
    possible; the first ``n_blocks % tp_size`` ranks receive one extra block.
    """
    assert n_blocks >= tp_size, (
        f"uneven block sharding needs n_blocks ({n_blocks}) >= tp_size ({tp_size})"
    )
    base, extra = divmod(n_blocks, tp_size)
    shards: list[tuple[int, int]] = []
    start = 0
    for rank in range(tp_size):
        nb = base + (1 if rank < extra else 0)
        shards.append((start, nb))
        start += nb
    assert start == n_blocks
    return shards


def block_n_from_quant_config(quant_config: QuantizationConfig | None) -> int | None:
    """Return the FP8 weight block size (block_n) if block-quantized, else None."""
    block_size = getattr(quant_config, "weight_block_size", None)
    if block_size is None or len(block_size) < 1:
        return None
    return int(block_size[0])


def should_use_uneven_block_sharding(
    intermediate_size: int, tp_size: int, block_n: int
) -> bool:
    """True iff the intermediate is block-aligned but not ``(tp*block)``-aligned.

    In that case the even split would place a 128-block across two ranks (which the
    shared FP8 block scale cannot represent); uneven whole-block sharding is required.
    Requires ``n_blocks >= tp_size`` so every rank gets at least one block.
    """
    if tp_size <= 1:
        return False
    if intermediate_size % block_n != 0:
        return False
    if intermediate_size % (tp_size * block_n) == 0:
        return False
    return (intermediate_size // block_n) >= tp_size


def _install_block_slicing_loaders(
    gate_up: MergedColumnParallelLinear,
    down: RowParallelLinear,
    block_start: int,
    n_local_blocks: int,
    block_n: int,
) -> None:
    """Replace the parameter weight loaders to copy this rank's block slice.

    The linears are built with ``disable_tp=True`` and this rank's *local* sizes, so
    each parameter is already correctly shaped; these loaders simply select the
    matching whole-block slice from the full checkpoint tensor. Weights are sliced in
    element units (``block_start * block_n``); block scales in grid units
    (``block_start``).
    """
    local_inter = n_local_blocks * block_n
    elem_off = block_start * block_n

    # These mirror the stock column/row parallel loaders exactly except for the
    # source offset (this rank's block range instead of tp_rank * shard_size).
    # In particular dtype handling is identical to the stock path: ``narrow`` is
    # dtype-agnostic and ``copy_`` performs any needed conversion (e.g. the
    # e8m0fnu block scales of FP4 checkpoints are numerically converted into the
    # float32 scale parameter, matching the even-TP path).

    # gate_up: MergedColumnParallelLinear, output (rows, dim 0). shard_id 0 -> gate
    # (w1), 1 -> up (w3); each writes into its half of the merged weight.
    def gate_up_weight_loader(param, loaded_weight, shard_id):
        sl = loaded_weight.narrow(0, elem_off, local_inter)
        param.data.narrow(0, shard_id * local_inter, local_inter).copy_(sl)

    def gate_up_scale_loader(param, loaded_weight, shard_id):
        sl = loaded_weight.narrow(0, block_start, n_local_blocks)
        param.data.narrow(0, shard_id * n_local_blocks, n_local_blocks).copy_(sl)

    # down_proj: RowParallelLinear, input (cols, dim 1).
    def down_weight_loader(param, loaded_weight):
        sl = loaded_weight.narrow(1, elem_off, local_inter)
        param.data.copy_(sl)

    def down_scale_loader(param, loaded_weight):
        sl = loaded_weight.narrow(1, block_start, n_local_blocks)
        param.data.copy_(sl)

    # BasevLLMParameter.weight_loader has a setter; overriding it is supported
    # (create_weights already installed the default v2 loader, so we replace it).
    gate_up.weight.weight_loader = gate_up_weight_loader
    gate_up.weight_scale_inv.weight_loader = gate_up_scale_loader
    down.weight.weight_loader = down_weight_loader
    down.weight_scale_inv.weight_loader = down_scale_loader


def make_block_sharded_shared_expert_linears(
    hidden_size: int,
    intermediate_size: int,
    quant_config: QuantizationConfig | None,
    block_n: int,
    prefix: str,
) -> tuple[MergedColumnParallelLinear, RowParallelLinear]:
    """Build ``gate_up_proj`` / ``down_proj`` for uneven block-aligned TP sharding.

    The linears use ``disable_tp=True`` with this rank's local block-multiple sizes;
    the per-parameter loaders select this rank's blocks from the checkpoint. The
    caller is responsible for the cross-rank reduction (all-reduce of the down_proj
    partial when ``reduce_results`` is desired).
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    n_blocks = intermediate_size // block_n
    block_start, n_local_blocks = compute_block_shards(n_blocks, tp_size)[tp_rank]
    local_inter = n_local_blocks * block_n

    gate_up = MergedColumnParallelLinear(
        hidden_size,
        [local_inter] * 2,
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
        prefix=f"{prefix}.gate_up_proj",
    )
    down = RowParallelLinear(
        local_inter,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
        prefix=f"{prefix}.down_proj",
    )
    _install_block_slicing_loaders(gate_up, down, block_start, n_local_blocks, block_n)
    return gate_up, down

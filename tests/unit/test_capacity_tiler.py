"""The capacity-aware intra-core tiler streams a resident weight on overflow, leaves fitting groups alone."""

import math
import os
import subprocess
import sys
import tempfile

import pytest

from stream.mapping.capacity_tiler import CapacityTiler, _divisors_desc, per_core_tiles
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

_TPU_QUAD = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
# A single large Gemm (K=8192) whose weight slice overflows the 2 MB matmul core when kept resident.
_GEMM = "stream/inputs/aie/workload/gemm_256_8192_2048.onnx"
# A SwiGLU small enough to fit the same cores without any streaming.
_SWIGLU_FITS = "stream/inputs/aie/workload/swiglu_256_512_2048.onnx"


def _parse(hardware: str, workload: str):
    ctx = StageContext.from_kwargs(accelerator=hardware, workload_path=workload, output_path=tempfile.mkdtemp())
    ctxs = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()
    return ctxs[0].get("accelerator"), ctxs[0].get("workload")


def _worst_core_ratio(gen: GenericMappingGenerator, sub, cns, tiling) -> float:
    """Worst per-core footprint / (capacity * fill) after applying ``tiling`` by global dim, using the
    same arithmetic footprint model the tiler does (full tensor / inter-core split, scaled by tiles)."""
    tiler = CapacityTiler(sub, gen.accelerator)
    unroll = gen._inter_core_unrolling(sub, cns)
    node_tensors = {cn: tiler._node_tensors(cn) for cn in cns}
    all_dims = {d for ts in node_tensors.values() for entry in ts for d in entry[1]}
    percore = {d: (sub.get_dimension_size(d) // unroll.get(d, 1)) for d in all_dims}
    resident = dict(percore)
    resident.update(tiler._seed_resident(cns, tiling, {d: 1 for d in all_dims}))

    def bits(t, dims):
        base = math.prod(t.shape) * t.operand_type.bitwidth / math.prod(unroll[d] for d in dims if unroll.get(d, 1) > 1)
        return base * math.prod(resident[d] / percore[d] for d in dims if percore.get(d, 0) > 0)

    caps: dict[int, float] = {}
    foot: dict[int, float] = {}
    seen: dict[int, set] = {}
    for cn in cns:
        for core in gen._select_cores_for_node(cn):
            caps[core.id] = core.get_memory_capacity()
            s = seen.setdefault(core.id, set())
            for tensor, dims in node_tensors[cn]:
                if tensor.name not in s:
                    s.add(tensor.name)
                    foot[core.id] = foot.get(core.id, 0.0) + bits(tensor, dims)
    return max((foot[c] / (caps[c] * tiler.fill_fraction) for c in caps if caps[c] > 0), default=0.0)


def test_divisors_desc():
    assert _divisors_desc(12) == [12, 6, 4, 3, 2, 1]
    assert _divisors_desc(1) == [1]
    assert _divisors_desc(14336)[0] == 14336  # sqrt enumeration returns the whole dim first


def test_streams_contraction_axis_when_weight_overflows():
    """A large Gemm whose resident weight overflows gets its contraction axis (D1) tiled until it fits."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    subs = w.split_fusion_groups(cut_points=gen._cut_points(None))
    refined_any = False
    for sub in subs:
        cns = tuple(sub.get_computation_nodes())
        if not cns:
            continue
        seed = gen._auto_fusion_tiling(sub, cns) or gen._whole_layer_tiling(sub, cns)
        refined = gen._capacity_refine(sub, cns, seed)
        if _worst_core_ratio(gen, sub, cns, seed) > 1.0:
            refined_any = True
            # the trivial mapper overflowed; the refined tiling must fit and must tile a contraction axis
            assert _worst_core_ratio(gen, sub, cns, refined) <= 1.0
            assert any(".D1" in e["dim"] for e in refined), f"expected a contraction-axis tile, got {refined}"
    assert refined_any, "the Gemm was expected to overflow the trivial mapping"


def test_noop_when_group_fits():
    """A group whose footprint already fits is returned unchanged -- no over-tiling."""
    acc, w = _parse(_TPU_QUAD, _SWIGLU_FITS)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    for sub in w.split_fusion_groups(cut_points=gen._cut_points(None)):
        cns = tuple(sub.get_computation_nodes())
        if not cns:
            continue
        seed = gen._auto_fusion_tiling(sub, cns) or gen._whole_layer_tiling(sub, cns)
        if _worst_core_ratio(gen, sub, cns, seed) <= 1.0:
            assert gen._capacity_refine(sub, cns, seed) == seed


def test_footprint_is_summed_per_physical_core():
    """A fused group is accounted per physical core: distinct tensors sum, a shared tensor counts once."""
    acc, w = _parse(_TPU_QUAD, _SWIGLU_FITS)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    sub = next(s for s in w.split_fusion_groups(cut_points=gen._cut_points(None)) if tuple(s.get_computation_nodes()))
    cns = tuple(sub.get_computation_nodes())
    assert len(cns) > 1, "need a multi-node fused group to observe summation"
    tiler = CapacityTiler(sub, acc)
    node_tensors = {cn: tiler._node_tensors(cn) for cn in cns}

    def bits(tensor) -> float:
        return math.prod(tensor.shape) * tensor.operand_type.bitwidth

    # A producer/consumer pair that shares an intermediate tensor, so dedup is observable.
    pair = next(
        (
            (a, b, shared)
            for a in cns
            for b in cns
            if a is not b and (shared := {t.name for t, _ in node_tensors[a]} & {t.name for t, _ in node_tensors[b]})
        ),
        None,
    )
    assert pair, "a fused group must have a producer/consumer pair sharing a tensor"
    a, b, shared = pair

    # The tiler's bucket for a core hosting both nodes: each tensor once, by name (its `core_seen` set).
    bucket: dict[str, float] = {}
    for cn in (a, b):
        for t, _ in node_tensors[cn]:
            bucket.setdefault(t.name, bits(t))
    packed_bits = sum(bucket.values())

    a_bits = sum(bits(t) for t, _ in node_tensors[a])
    b_bits = sum(bits(t) for t, _ in node_tensors[b])
    shared_bits = sum(bits(t) for t, _ in node_tensors[a] if t.name in shared)

    # SUM, with the shared intermediate counted once rather than twice.
    assert packed_bits == pytest.approx(a_bits + b_bits - shared_bits)
    assert packed_bits > max(a_bits, b_bits)  # the shared core holds more than either node alone
    assert packed_bits < a_bits + b_bits  # ... but less than a per-view sum that double-counts it


def test_refined_mapping_validates():
    """The mapping the refined tiling produces still passes MappingValidator."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    paths, _ = gen.generate_all_groups()
    for path in paths:
        import yaml

        data = yaml.safe_load(open(path))
        assert MappingValidator(data).validate(), MappingValidator(data).errors


def test_mapping_independent_of_hash_seed():
    """Tied dims are ordered by position, not by their seed-dependent hash, so every process maps alike."""
    script = (
        "import tempfile, sys\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        "from test_capacity_tiler import _SWIGLU_FITS, _TPU_QUAD, _parse\n"
        "from stream.mapping.generic_generator import GenericMappingGenerator\n"
        "acc, w = _parse(_TPU_QUAD, _SWIGLU_FITS)\n"
        "for path in GenericMappingGenerator(acc, w, tempfile.mkdtemp()).generate_all_groups()[0]:\n"
        "    print(open(path).read())\n"
    )
    mappings = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1")
    }
    assert len(mappings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_per_core_tiles_of_an_uneven_split():
    """4096 over 3 cores: each temporal step that divides 4096 hands the busiest core a ceiling tile."""
    assert per_core_tiles(4096, 3)[:4] == [1366, 683, 342, 171]
    assert per_core_tiles(4096, 4) == [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

"""A lone memory-bound node must not be reported faster than its weights can stream from off-chip.

`estimated_cycles` returns the pipelined throughput bound for a single-node group. That bound once
took the max busy time over compute cores only, so a GEMV whose weights cross the HBM link reported
a third of the time the link physically needs.
"""

import tempfile
from pathlib import Path

from onnx import TensorProto, helper, save

from stream.api import evaluate_mapping

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
HBM_BITS_PER_CYCLE = 107345


def _gemm(m: int, n: int, k: int, out_dir: str) -> str:
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W"], ["Y"], name="Gemm")],
        "gemm",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [m, n])],
        initializer=[TensorProto(name="W", data_type=TensorProto.BFLOAT16, dims=[k, n])],
    )
    path = str(Path(out_dir) / "gemm.onnx")
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_memory_bound_gemm_respects_offchip_bandwidth():
    m, n, k = 16, 8192, 8192
    with tempfile.TemporaryDirectory() as tmp:
        estimate = evaluate_mapping(TPU_V7, _gemm(m, n, k, tmp), tmp)
    weight_stream_cycles = n * k * 16 / HBM_BITS_PER_CYCLE
    assert estimate.cycles >= weight_stream_cycles

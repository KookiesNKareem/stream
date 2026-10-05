"""Two TensorCores, each on its own bus, against the same cores on one bus of their combined bandwidth.

A weight-streaming GEMV is bound by HBM either way. Splitting the bus once doubled it: the planner
routed every memory tile to every compute core, so the half of the weights each TensorCore never
reads crossed to the other one over the HBM bus, and the two buses, equal in bandwidth, were taken
for one link.
"""

import tempfile
from pathlib import Path

import yaml
from onnx import TensorProto, helper, save

from stream.api import evaluate_mapping
from stream.stages.parsing.accelerator_parser import parse_accelerator

CORES = Path(__file__).resolve().parents[1] / "stream/inputs/examples/hardware/cores"
TC_BUS = 416974
HBM_BUS = 53672
D2D = 208487
TCS = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))  # four MXUs, then the VMEM
HBM = 10


def _hardware(out_dir: str, per_tc: bool) -> str:
    mxu, vmem, hbm = (yaml.safe_load((CORES / f"tpu_v7_{name}.yaml").read_text()) for name in ("mxu", "vmem", "hbm"))
    cores, coords = {}, {}
    for col, tc in enumerate(TCS):
        for row, core in enumerate(tc):
            cores[core] = vmem if core == tc[-1] else mxu
            coords[core] = [col, row]
    cores[HBM] = hbm
    coords[HBM] = [0, len(TCS[0])]
    vmems = [tc[-1] for tc in TCS]
    if per_tc:
        buses = [{"type": "bus", "cores": list(tc), "bandwidth": TC_BUS} for tc in TCS]
    else:
        buses = [{"type": "bus", "cores": [c for tc in TCS for c in tc], "bandwidth": 2 * TC_BUS}]
    system = {
        "name": "two_tc",
        "cores": cores,
        "core_coordinates": coords,
        "offchip_core_id": HBM,
        "unit_energy_cost": 0,
        "core_connectivity": [
            *buses,
            {"type": "link", "cores": vmems, "bandwidth": D2D},
            {"type": "bus", "cores": [*vmems, HBM], "bandwidth": HBM_BUS},
        ],
    }
    path = Path(out_dir) / f"two_tc_{'per_tc' if per_tc else 'global'}.yaml"
    path.write_text(yaml.safe_dump(system))
    return str(path)


def _gemv(n: int, k: int, out_dir: str) -> str:
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W"], ["Y"], name="Gemm")],
        "gemv",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, k])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, n])],
        initializer=[TensorProto(name="W", data_type=TensorProto.BFLOAT16, dims=[k, n])],
    )
    path = str(Path(out_dir) / "gemv.onnx")
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_each_memory_tile_feeds_only_its_own_cores_over_its_own_bus():
    with tempfile.TemporaryDirectory() as tmp:
        accelerator = parse_accelerator(_hardware(tmp, per_tc=True))
    core = accelerator.get_core
    plans = accelerator.communication_manager.get_possible_transfer_plan(
        [core(tc[-1]) for tc in TCS], [core(c) for tc in TCS for c in tc[:-1]]
    )
    for plan in plans:
        assert sorted(link.members for link in plan.links_used) == [TCS[0], TCS[1]]
        assert plan.link_shares() == (0.5, 0.5)


def test_overlapping_connections_route_over_the_widest():
    with tempfile.TemporaryDirectory() as tmp:
        accelerator = parse_accelerator(_hardware(tmp, per_tc=True))
    vmem0, vmem1 = (accelerator.get_core(tc[-1]) for tc in TCS)
    assert accelerator.cores.edges[(vmem0, vmem1)]["cl"].bandwidth == D2D


def test_per_tensorcore_buses_match_one_bus_of_their_combined_bandwidth():
    n = k = 4096
    with tempfile.TemporaryDirectory() as tmp:
        onnx = _gemv(n, k, tmp)
        per_tc = evaluate_mapping(_hardware(tmp, per_tc=True), onnx, f"{tmp}/per_tc").cycles
        one_bus = evaluate_mapping(_hardware(tmp, per_tc=False), onnx, f"{tmp}/global").cycles
    floor = n * k * 16 / HBM_BUS
    assert one_bus >= floor
    assert abs(per_tc - one_bus) <= 0.05 * one_bus

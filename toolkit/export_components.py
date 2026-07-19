"""Exporta los 3 grafos "faciles" de GMFSS_Fortuna (variante base/PG, sin IFNet) a ONNX
opset 17, y guarda tensores de validacion reales (derivados/cargados de refs/golden/,
NUNCA randn) junto a cada .onnx en artifacts/.

Patron estructural tomado de port-audiosr-onnx/toolkit/export_components.py: exportar con
el legacy JIT exporter (dynamo=False) primero; si un grafo tripea el tracer, reintentar con
dynamo=True/opset 18 -- la parity decide, no una preferencia (ver validate_ort.py).

Grafos (variante "base" -- ver docs/vendored-sources.md; NO incluye IFNet/RIFE, eso es
solo de la variante "union" que este port no usa):
  featurenet.onnx  FeatureNet.forward(img) -> (scale1, scale2, scale3)
  metricnet.onnx   MetricNet.forward(img0_half, img1_half, flow01, flow10) -> (metric0, metric1)
  fusionnet.onnx   GridNet.forward(fusion_rgb, fusion_feat1, fusion_feat2, fusion_feat3) -> raw_out
  gmflow.onnx      GMFlow.forward(img0_half, img1_half) -> flow  (attn/corr/prop-radius lists y
                   pred_bidir_flow=False horneados fijos como en gmfss_pg_pipeline.py; se llama
                   dos veces en runtime con args swapeados para flow01/flow10 usando el mismo grafo)

Shapes fijos a la resolucion objetivo 1920x1088 (1080p pad /64) -- sin dynamic_axes.

Uso: .venv/Scripts/python.exe toolkit/export_components.py [featurenet metricnet fusionnet gmflow]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TOOLKIT_DIR = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
GOLDEN_DIR = ROOT / "refs" / "golden"
MODELS_DIR = TOOLKIT_DIR / "vendor" / "vs_gmfss_fortuna" / "models"

sys.path.insert(0, str(TOOLKIT_DIR))
from gmfss_pg_pipeline import _load_state_dict, resize_bilinear  # noqa: E402

sys.path.insert(0, str(TOOLKIT_DIR / "vendor"))
from vs_gmfss_fortuna.FeatureNet import FeatureNet  # noqa: E402
from vs_gmfss_fortuna.FusionNet_b import GridNet  # noqa: E402
from vs_gmfss_fortuna.gmflow.gmflow import GMFlow  # noqa: E402
from vs_gmfss_fortuna.MetricNet import MetricNet  # noqa: E402

ART.mkdir(exist_ok=True)

OPSET = 17
PRIMARY_PAIR = "vf_t006"
EXTRA_VALIDATION_PAIRS = ("vs_t013", "vwarm_t019")


def discover_pairs() -> list[str]:
    ordered = [PRIMARY_PAIR, *EXTRA_VALIDATION_PAIRS]
    return [name for name in ordered if (GOLDEN_DIR / name).is_dir()]


def load_golden(pair: str, name: str) -> torch.Tensor:
    array = np.load(GOLDEN_DIR / pair / f"{name}.npy")
    return torch.from_numpy(array)


def half_size(tensor: torch.Tensor) -> tuple[int, int]:
    return tensor.shape[2] // 2, tensor.shape[3] // 2


def derive_img_half(pair: str, which: str) -> torch.Tensor:
    """img{0,1}_half no se dumpeo en fase 0 -- es F.interpolate exacto y reproducible
    sobre input_norm_img{0,1}, igual que gmfss_pg_pipeline.reuse()/forward()."""
    img = load_golden(pair, f"input_norm_img{which}")
    return resize_bilinear(img, *half_size(img))


def save_case(name: str, case_id: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> None:
    for i, tensor in enumerate(inputs):
        np.save(ART / f"{name}_{case_id}_in{i}.npy", tensor.detach().cpu().numpy())
    for i, tensor in enumerate(outputs):
        np.save(ART / f"{name}_{case_id}_ref{i}.npy", tensor.detach().cpu().numpy())


def as_tuple(value: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def load_golden_outputs(pair: str, names: list[str]) -> list[torch.Tensor]:
    """Carga los tensores de salida YA DUMPEADOS por fase 0 (run_reference.py) en
    refs/golden/<pair>/*.npy -- estas son las refs de verdad para el gate de parity,
    NO una pasada eager fresca de este script (ver Finding 1 del review de Task 1.1:
    comparar el ONNX contra el propio eager de este script no prueba nada sobre bugs
    de carga de pesos/estado-eval, porque ambos lados heredarian el mismo bug)."""
    return [load_golden(pair, n) for n in names]


EAGER_VS_GOLDEN_WARN_THRESHOLD = 1e-3


def rel_err(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = np.abs(expected).max()
    if denom == 0:
        return float(np.abs(actual - expected).max())
    return float(np.abs(actual - expected).max() / denom)


def warn_if_eager_diverges_from_golden(
    name: str,
    case_id: str,
    eager_outputs: tuple[torch.Tensor, ...],
    golden_refs: list[torch.Tensor],
) -> None:
    """Sanity check independiente del gate de parity en validate_ort.py: compara la pasada
    eager de ESTE script contra el output dorado de fase 0. Si diverge por encima del
    umbral, es senal de un bug real (pesos, estado eval, version de modulo) -- se imprime,
    no se silencia ni se usa para gatear el export."""
    for i, (eager, golden) in enumerate(zip(eager_outputs, golden_refs)):
        err = rel_err(eager.detach().cpu().numpy(), golden.detach().cpu().numpy())
        status = "OK" if err < EAGER_VS_GOLDEN_WARN_THRESHOLD else "MISMATCH"
        print(f"[export]   eager-vs-golden {name}/{case_id}/out{i}: rel-err={err:.6e} [{status}]", flush=True)


def export_graph(
    module: nn.Module,
    name: str,
    inputs: list[torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    refs: list[torch.Tensor],
    dynamo: bool = False,
) -> None:
    """Exporta `module` a artifacts/<name>.onnx usando `inputs` como el caso "case0"
    (el trace concreto que queda horneado en el grafo), y guarda ese caso -- con `refs`
    (los outputs dorados REALES de fase 0, no una pasada eager fresca) -- como par de
    validacion real para validate_ort.py."""
    path = ART / f"{name}.onnx"
    module.eval()
    with torch.no_grad():
        eager_outputs = as_tuple(module(*inputs))
    warn_if_eager_diverges_from_golden(name, "case0", eager_outputs, refs)
    save_case(name, "case0", inputs, refs)
    torch.onnx.export(
        module,
        tuple(inputs),
        str(path),
        opset_version=18 if dynamo else OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamo=dynamo,
    )
    shapes = [tuple(r.shape) for r in refs]
    size_mb = path.stat().st_size / 1e6
    print(f"[export] {name}: {size_mb:.1f} MB, out shapes {shapes}, dynamo={dynamo}", flush=True)


def add_extra_case(
    module: nn.Module,
    name: str,
    case_id: str,
    inputs: list[torch.Tensor],
    refs: list[torch.Tensor],
) -> None:
    """Guarda un caso REAL adicional (otro par de refs/golden/, u otra rama del mismo par)
    contra su output dorado de fase 0 (`refs`), para que validate_ort.py verifique el .onnx
    exportado contra mas de un tensor real. Corre el modulo eager solo como sanity check
    diagnostico (warn_if_eager_diverges_from_golden) -- no se guarda ni se usa como ref."""
    module.eval()
    with torch.no_grad():
        eager_outputs = as_tuple(module(*inputs))
    warn_if_eager_diverges_from_golden(name, case_id, eager_outputs, refs)
    save_case(name, case_id, inputs, refs)
    print(f"[export]   + extra case {case_id} for {name}", flush=True)


def build_featurenet() -> FeatureNet:
    net = FeatureNet()
    net.load_state_dict(_load_state_dict(MODELS_DIR / "feat_base.pkl"))
    net.eval()
    return net


def build_metricnet() -> MetricNet:
    net = MetricNet()
    net.load_state_dict(_load_state_dict(MODELS_DIR / "metric_base.pkl"))
    net.eval()
    return net


def build_fusionnet() -> GridNet:
    net = GridNet()
    net.load_state_dict(_load_state_dict(MODELS_DIR / "fusionnet_base.pkl"))
    net.eval()
    return net


class GMFlowExportWrapper(nn.Module):
    """Fija a inputs los defaults que gmfss_pg_pipeline.py usa en la llamada real
    (`self.flownet(img0_half, img1_half)`, sin kwargs) -- attn_splits_list, corr_radius_list y
    prop_radius_list quedan horneados como listas Python constantes, no como inputs del grafo,
    y pred_bidir_flow=False elimina la rama de batch duplicado. El mismo grafo se llama dos
    veces en runtime con img0/img1 swapeados para producir flow01 y flow10."""

    def __init__(self, flownet: GMFlow) -> None:
        super().__init__()
        self.flownet = flownet

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        return self.flownet(
            img0,
            img1,
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            pred_bidir_flow=False,
        )


def build_gmflow() -> GMFlowExportWrapper:
    net = GMFlow()
    net.load_state_dict(_load_state_dict(MODELS_DIR / "flownet.pkl"))
    net.eval()
    wrapper = GMFlowExportWrapper(net)
    wrapper.eval()
    return wrapper


def featurenet_output_names(which: str) -> list[str]:
    prefix = "feat0" if which == "0" else "feat1"
    return [f"{prefix}_scale1", f"{prefix}_scale2", f"{prefix}_scale3"]


def export_featurenet(pairs: list[str]) -> None:
    net = build_featurenet()
    primary = pairs[0]
    img0 = load_golden(primary, "input_norm_img0")
    refs0 = load_golden_outputs(primary, featurenet_output_names("0"))
    export_graph(net, "featurenet", [img0], ["img"], ["scale1", "scale2", "scale3"], refs0)

    # img1 del mismo par: misma resolucion, mismo grafo, tensor real distinto.
    img1 = load_golden(primary, "input_norm_img1")
    refs1 = load_golden_outputs(primary, featurenet_output_names("1"))
    add_extra_case(net, "featurenet", "case1", [img1], refs1)

    case_id = 2
    for pair in pairs[1:]:
        for which in ("0", "1"):
            img = load_golden(pair, f"input_norm_img{which}")
            refs = load_golden_outputs(pair, featurenet_output_names(which))
            add_extra_case(net, "featurenet", f"case{case_id}", [img], refs)
            case_id += 1


def build_metricnet_inputs(pair: str) -> list[torch.Tensor]:
    return [
        derive_img_half(pair, "0"),
        derive_img_half(pair, "1"),
        load_golden(pair, "flow01"),
        load_golden(pair, "flow10"),
    ]


def export_metricnet(pairs: list[str]) -> None:
    net = build_metricnet()
    primary = pairs[0]
    # legacy JIT exporter (opset 17) trips on aten::l1_loss ("Exporting the operator
    # 'aten::l1_loss' to ONNX opset version 17 is not supported") -- dynamo=True/opset 18
    # handles it. Parity decides, not preference (see validate_ort.py rel-err gate).
    export_graph(
        net,
        "metricnet",
        build_metricnet_inputs(primary),
        ["img0_half", "img1_half", "flow01", "flow10"],
        ["metric0", "metric1"],
        load_golden_outputs(primary, ["metric0", "metric1"]),
        dynamo=True,
    )
    for i, pair in enumerate(pairs[1:], start=1):
        refs = load_golden_outputs(pair, ["metric0", "metric1"])
        add_extra_case(net, "metricnet", f"case{i}", build_metricnet_inputs(pair), refs)


def build_fusionnet_inputs(pair: str) -> list[torch.Tensor]:
    img0_half = derive_img_half(pair, "0")
    img1_half = derive_img_half(pair, "1")
    splat_i1t = load_golden(pair, "splat_I1t")
    splat_i2t = load_golden(pair, "splat_I2t")
    fusion_rgb = torch.cat([img0_half, splat_i1t, splat_i2t, img1_half], dim=1)

    fusion_feat1 = torch.cat(
        [load_golden(pair, "splat_feat1t1"), load_golden(pair, "splat_feat2t1")], dim=1
    )
    fusion_feat2 = torch.cat(
        [load_golden(pair, "splat_feat1t2"), load_golden(pair, "splat_feat2t2")], dim=1
    )
    fusion_feat3 = torch.cat(
        [load_golden(pair, "splat_feat1t3"), load_golden(pair, "splat_feat2t3")], dim=1
    )
    return [fusion_rgb, fusion_feat1, fusion_feat2, fusion_feat3]


def export_fusionnet(pairs: list[str]) -> None:
    net = build_fusionnet()
    primary = pairs[0]
    export_graph(
        net,
        "fusionnet",
        build_fusionnet_inputs(primary),
        ["fusion_rgb", "fusion_feat1", "fusion_feat2", "fusion_feat3"],
        ["raw_out"],
        load_golden_outputs(primary, ["fusionnet_out"]),
    )
    for i, pair in enumerate(pairs[1:], start=1):
        refs = load_golden_outputs(pair, ["fusionnet_out"])
        add_extra_case(net, "fusionnet", f"case{i}", build_fusionnet_inputs(pair), refs)


def build_gmflow_case(pair: str, direction: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """direction '01': flownet(img0_half, img1_half) -> flow01 (golden ref ya dumpeado en
    fase 0). direction '10': mismo grafo, args swapeados -> flow10, exactamente como
    gmfss_pg_pipeline.estimate_flow_and_metric() llama flownet() dos veces."""
    if direction == "01":
        inputs = [derive_img_half(pair, "0"), derive_img_half(pair, "1")]
        refs = load_golden_outputs(pair, ["flow01"])
    else:
        inputs = [derive_img_half(pair, "1"), derive_img_half(pair, "0")]
        refs = load_golden_outputs(pair, ["flow10"])
    return inputs, refs


def export_gmflow(pairs: list[str]) -> None:
    net = build_gmflow()
    primary = pairs[0]
    inputs0, refs0 = build_gmflow_case(primary, "01")
    export_graph(net, "gmflow", inputs0, ["img0_half", "img1_half"], ["flow"], refs0)

    inputs1, refs1 = build_gmflow_case(primary, "10")
    add_extra_case(net, "gmflow", "case1", inputs1, refs1)

    case_id = 2
    for pair in pairs[1:]:
        for direction in ("01", "10"):
            inputs, refs = build_gmflow_case(pair, direction)
            add_extra_case(net, "gmflow", f"case{case_id}", inputs, refs)
            case_id += 1


GRAPH_EXPORTERS = {
    "featurenet": export_featurenet,
    "metricnet": export_metricnet,
    "fusionnet": export_fusionnet,
    "gmflow": export_gmflow,
}


def main() -> None:
    requested = sys.argv[1:] or list(GRAPH_EXPORTERS)
    pairs = discover_pairs()
    if not pairs:
        raise SystemExit(f"no golden pairs found in {GOLDEN_DIR}")
    print(f"[export] golden pairs: {pairs}", flush=True)

    for graph_name in requested:
        exporter = GRAPH_EXPORTERS[graph_name]
        exporter(pairs)

    print("[export] done", flush=True)


if __name__ == "__main__":
    main()

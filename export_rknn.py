#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path
from typing import Any, Sequence


ANALYSIS_OUTPUTS = ("latent", "hyper_latent", "scales_y", "means_y")
PINNED_PREFIX = "rknn_keep"


def check_ret(ret: object, step: str) -> None:
    if ret is not None and ret != 0:
        raise SystemExit(f"{step} failed with code {ret}")


def split_csv(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in values:
        expanded.extend(part.strip() for part in str(item).split(",") if part.strip())
    return expanded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a fixed-size hyper analysis ONNX encoder to RKNN. "
            "The default analysis path pins y,z,scales_y,means_y as terminal "
            "graph outputs before RKNN conversion so the exported RKNN keeps "
            "the four outputs required by the board compressor."
        )
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Input ONNX model.")
    parser.add_argument("--output", type=Path, default=None, help="Output RKNN model.")
    parser.add_argument(
        "--part",
        choices=("analysis", "generic"),
        default="analysis",
        help="analysis expects latent,hyper_latent,scales_y,means_y outputs.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8", "mixed"),
        default="fp16",
        help="fp16 is recommended for hyper analysis. int8/mixed require --dataset.",
    )
    parser.add_argument("--input-name", default="input", help="ONNX input tensor name.")
    parser.add_argument(
        "--output-name",
        action="append",
        default=[],
        help=(
            "Analysis ONNX output tensor name. Repeat or comma-separate. "
            "Default: latent,hyper_latent,scales_y,means_y."
        ),
    )
    parser.add_argument("--width", type=int, default=1024, help="Model input width.")
    parser.add_argument("--height", type=int, default=1024, help="Model input height.")
    parser.add_argument("--channels-y", type=int, default=192, help="Latent y channels.")
    parser.add_argument("--channels-z", type=int, default=128, help="Hyper latent z channels.")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform.")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=None, help="Quantization dataset text file.")
    parser.add_argument(
        "--strategy",
        choices=(
            "pinned-no-crop",
            "pinned-node",
            "pinned-tensor",
            "no-crop",
            "producer-node",
            "tensor",
        ),
        default="pinned-no-crop",
        help=(
            "How to preserve/select analysis outputs. pinned-no-crop is the "
            "recommended route because RKNN does not receive an outputs crop list."
        ),
    )
    parser.add_argument(
        "--pinned-onnx",
        type=Path,
        default=None,
        help="Path for the temporary pinned ONNX used by pinned-* strategies.",
    )
    parser.add_argument(
        "--keep-pinned-onnx",
        action="store_true",
        help="Keep the generated pinned ONNX next to --output for inspection.",
    )
    parser.add_argument(
        "--force-input-size",
        action="store_true",
        help=(
            "Pass inputs/input_size_list to RKNN. Leave off for static-shape ONNX "
            "exports; forcing input size also enables RKNN crop mode."
        ),
    )
    parser.add_argument(
        "--allow-shape-warnings",
        action="store_true",
        help="Warn instead of exiting when ONNX shapes do not match width/height.",
    )
    parser.add_argument("--list-io", action="store_true", help="Print ONNX IO names/shapes and exit.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be > 0")
    if args.channels_y <= 0 or args.channels_z <= 0:
        raise SystemExit("--channels-y and --channels-z must be > 0")
    if args.part == "analysis" and (args.width % 64 != 0 or args.height % 64 != 0):
        raise SystemExit("analysis width/height must be divisible by 64")
    if args.precision in {"int8", "mixed"} and args.dataset is None:
        raise SystemExit("--dataset is required when --precision int8/mixed is used")

    args.output_name = split_csv(args.output_name)
    if args.part == "analysis" and not args.output_name:
        args.output_name = list(ANALYSIS_OUTPUTS)
    if args.part == "analysis" and len(args.output_name) != 4:
        raise SystemExit("analysis conversion requires exactly four --output-name values")
    if args.part != "analysis" and args.strategy == "pinned-no-crop":
        args.strategy = "no-crop"
    if args.part != "analysis" and args.strategy.startswith("pinned"):
        raise SystemExit("pinned-* strategies are only valid with --part analysis")
    if not args.list_io and args.output is None:
        raise SystemExit("--output is required unless --list-io is used")
    return args


def import_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "onnx is required for this converter because analysis outputs are "
            "validated and pinned before RKNN conversion. Install it in the "
            "RKNN environment with: python -m pip install onnx"
        ) from exc
    return onnx


def load_onnx_model(onnx_path: Path) -> tuple[Any, Any]:
    onnx = import_onnx()
    model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise SystemExit(f"ONNX checker failed for {onnx_path}: {exc}") from exc
    return onnx, model


def tensor_shape(value_info: object) -> list[int | str | None]:
    tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
    shape = getattr(tensor_type, "shape", None)
    dims: list[int | str | None] = []
    for dim in getattr(shape, "dim", []):
        if hasattr(dim, "HasField") and dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif hasattr(dim, "HasField") and dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dim_value = getattr(dim, "dim_value", 0)
            dim_param = getattr(dim, "dim_param", "")
            dims.append(int(dim_value) if dim_value else (str(dim_param) if dim_param else None))
    return dims


def shape_text(shape: Sequence[int | str | None]) -> str:
    return "[" + ",".join("?" if dim is None else str(dim) for dim in shape) + "]"


def concrete_shape(shape: Sequence[int | str | None]) -> list[int] | None:
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        return None
    return [int(dim) for dim in shape]


def graph_output_map(model: object) -> dict[str, object]:
    return {value_info.name: value_info for value_info in model.graph.output}


def graph_input_map(model: object) -> dict[str, object]:
    return {value_info.name: value_info for value_info in model.graph.input}


def producer_map(model: object) -> dict[str, str]:
    producers: dict[str, str] = {}
    for node in model.graph.node:
        producer_name = (node.name or node.output[0]) if node.output else node.op_type
        for output in node.output:
            producers[output] = producer_name
    return producers


def expected_analysis_shapes(args: argparse.Namespace) -> list[list[int]]:
    y_h = int(args.height) // 16
    y_w = int(args.width) // 16
    z_h = int(args.height) // 64
    z_w = int(args.width) // 64
    return [
        [1, int(args.channels_y), y_h, y_w],
        [1, int(args.channels_z), z_h, z_w],
        [1, int(args.channels_y), y_h, y_w],
        [1, int(args.channels_y), y_h, y_w],
    ]


def warn_or_exit(message: str, allow_warning: bool) -> None:
    if allow_warning:
        print(f"[warn] {message}")
        return
    raise SystemExit(message + " Use --allow-shape-warnings to continue anyway.")


def validate_analysis_model(model: object, args: argparse.Namespace) -> dict[str, str]:
    outputs = graph_output_map(model)
    output_names = [value_info.name for value_info in model.graph.output]
    missing = [name for name in args.output_name if name not in outputs]
    if missing:
        raise SystemExit(
            "analysis ONNX must expose the four board outputs. "
            f"Missing: {', '.join(missing)}. Actual graph outputs: {', '.join(output_names)}"
        )

    inputs = graph_input_map(model)
    input_info = inputs.get(args.input_name)
    if input_info is None:
        actual_inputs = ", ".join(inputs) or "<none>"
        warn_or_exit(
            f"input {args.input_name!r} was not found. Actual graph inputs: {actual_inputs}",
            bool(args.allow_shape_warnings),
        )
    else:
        actual_input_shape = concrete_shape(tensor_shape(input_info))
        expected_input_shape = [1, 3, int(args.height), int(args.width)]
        if actual_input_shape is not None and actual_input_shape != expected_input_shape:
            warn_or_exit(
                f"input shape mismatch for {args.input_name}: got "
                f"{shape_text(actual_input_shape)}, expected {shape_text(expected_input_shape)}",
                bool(args.allow_shape_warnings),
            )

    expected_shapes = expected_analysis_shapes(args)
    for index, (name, expected) in enumerate(zip(args.output_name, expected_shapes)):
        actual = concrete_shape(tensor_shape(outputs[name]))
        if actual is None:
            print(f"[warn] output {index} {name}: dynamic/unknown shape {shape_text(tensor_shape(outputs[name]))}")
            continue
        if actual != expected:
            warn_or_exit(
                f"output {index} {name} shape mismatch: got {shape_text(actual)}, "
                f"expected {shape_text(expected)}",
                bool(args.allow_shape_warnings),
            )

    producers = producer_map(model)
    selected_producers: dict[str, str] = {}
    print("analysis_expected_io:")
    print(f"  input: {args.input_name} {shape_text([1, 3, int(args.height), int(args.width)])}")
    for index, (name, expected) in enumerate(zip(args.output_name, expected_shapes)):
        producer = producers.get(name, "")
        selected_producers[name] = producer
        display_producer = producer or "<graph-input-or-unknown>"
        print(f"  output{index}: {name} {shape_text(expected)} producer={display_producer}")
    return selected_producers


def list_onnx_io(onnx_path: Path) -> None:
    _, model = load_onnx_model(onnx_path)
    producers = producer_map(model)
    print("== inputs ==")
    for value_info in model.graph.input:
        print(f"{value_info.name} {shape_text(tensor_shape(value_info))}")
    print("== outputs ==")
    for value_info in model.graph.output:
        producer = producers.get(value_info.name, "<graph-input-or-unknown>")
        print(f"{value_info.name} {shape_text(tensor_shape(value_info))} producer={producer}")


def safe_suffix(name: str) -> str:
    suffix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return suffix.strip("_") or "output"


def make_pinned_output_names(output_names: Sequence[str]) -> tuple[list[str], list[str]]:
    tensors: list[str] = []
    nodes: list[str] = []
    for index, name in enumerate(output_names):
        suffix = safe_suffix(name)
        tensors.append(f"{PINNED_PREFIX}_{index}_{suffix}")
        nodes.append(f"{PINNED_PREFIX}_{index}_{suffix}_node")
    return tensors, nodes


def make_pinned_onnx(
    onnx: Any,
    model: object,
    output_names: Sequence[str],
    pinned_path: Path,
) -> tuple[list[str], list[str]]:
    pinned_model = copy.deepcopy(model)
    outputs = graph_output_map(pinned_model)
    pinned_tensors, pinned_nodes = make_pinned_output_names(output_names)

    new_graph_outputs = []
    for source_name, pinned_tensor, pinned_node in zip(output_names, pinned_tensors, pinned_nodes):
        source_info = outputs[source_name]
        identity = onnx.helper.make_node(
            "Identity",
            inputs=[source_name],
            outputs=[pinned_tensor],
            name=pinned_node,
        )
        pinned_model.graph.node.append(identity)
        pinned_info = copy.deepcopy(source_info)
        pinned_info.name = pinned_tensor
        new_graph_outputs.append(pinned_info)

    del pinned_model.graph.output[:]
    pinned_model.graph.output.extend(new_graph_outputs)
    onnx.checker.check_model(pinned_model)
    pinned_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(pinned_model, str(pinned_path))
    return pinned_tensors, pinned_nodes


def pinned_onnx_path(args: argparse.Namespace, temp_dir: Path | None) -> Path:
    if args.pinned_onnx is not None:
        return args.pinned_onnx
    if args.keep_pinned_onnx:
        return args.output.with_suffix(".pinned.onnx")
    if temp_dir is None:
        raise RuntimeError("internal error: temp_dir is required for temporary pinned ONNX")
    return temp_dir / f"{args.output.stem}.pinned.onnx"


def conversion_inputs(
    args: argparse.Namespace,
    onnx: Any,
    model: object,
    producers: dict[str, str],
    temp_dir: Path | None,
) -> tuple[Path, list[str] | None]:
    if args.part != "analysis":
        return args.onnx, split_csv(args.output_name)

    if args.strategy == "no-crop":
        return args.onnx, None
    if args.strategy == "producer-node":
        selected = [producers.get(name, "") for name in args.output_name]
        missing = [name for name, producer in zip(args.output_name, selected) if not producer]
        if missing:
            raise SystemExit(f"producer-node strategy could not find producers for: {', '.join(missing)}")
        return args.onnx, selected
    if args.strategy == "tensor":
        return args.onnx, list(args.output_name)

    pinned_path = pinned_onnx_path(args, temp_dir)
    pinned_tensors, pinned_nodes = make_pinned_onnx(onnx, model, args.output_name, pinned_path)
    print(f"pinned_onnx: {pinned_path.resolve()}")

    if args.strategy == "pinned-no-crop":
        return pinned_path, None
    if args.strategy == "pinned-node":
        return pinned_path, pinned_nodes
    if args.strategy == "pinned-tensor":
        return pinned_path, pinned_tensors
    raise SystemExit(f"unsupported strategy: {args.strategy}")


def build_rknn(args: argparse.Namespace, model_path: Path, rknn_outputs: list[str] | None) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit("rknn-toolkit2 is required: pip install rknn-toolkit2") from exc

    if args.dataset is not None and not args.dataset.exists():
        raise SystemExit(f"dataset file not found: {args.dataset}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=bool(args.verbose))
    try:
        check_ret(
            rknn.config(
                target_platform=args.target_platform,
                optimization_level=int(args.optimization_level),
                float_dtype="float16",
            ),
            "rknn.config",
        )
        load_kwargs: dict[str, object] = {"model": str(model_path)}
        if args.force_input_size:
            load_kwargs["inputs"] = [args.input_name]
            load_kwargs["input_size_list"] = [[1, 3, int(args.height), int(args.width)]]
        if rknn_outputs:
            load_kwargs["outputs"] = rknn_outputs
        print(f"strategy: {args.strategy}")
        print(f"load_onnx_kwargs: {load_kwargs}")
        check_ret(rknn.load_onnx(**load_kwargs), "rknn.load_onnx")

        if args.precision == "fp16":
            build_kwargs: dict[str, object] = {"do_quantization": False}
        elif args.precision == "mixed":
            build_kwargs = {"do_quantization": True, "dataset": str(args.dataset), "auto_hybrid": True}
        else:
            build_kwargs = {"do_quantization": True, "dataset": str(args.dataset)}
        check_ret(rknn.build(**build_kwargs), "rknn.build")
        check_ret(rknn.export_rknn(str(args.output)), "rknn.export_rknn")
    finally:
        rknn.release()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.onnx.exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    if args.list_io:
        list_onnx_io(args.onnx)
        return 0

    onnx, model = load_onnx_model(args.onnx)
    producers: dict[str, str] = {}
    if args.part == "analysis":
        producers = validate_analysis_model(model, args)
    else:
        producers = producer_map(model)

    if args.keep_pinned_onnx and args.pinned_onnx is not None:
        print("[warn] --pinned-onnx already keeps the generated pinned ONNX; --keep-pinned-onnx is redundant")

    with tempfile.TemporaryDirectory(prefix="rknn_pinned_") as temp_name:
        temp_dir = None if args.keep_pinned_onnx or args.pinned_onnx is not None else Path(temp_name)
        model_path, rknn_outputs = conversion_inputs(args, onnx, model, producers, temp_dir)
        build_rknn(args, model_path, rknn_outputs)

    print(f"Exported RKNN: {args.output.resolve()}")
    print("expected_board_rknn_outputs:")
    for index, shape in enumerate(expected_analysis_shapes(args) if args.part == "analysis" else []):
        print(f"  output{index}: {shape_text(shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

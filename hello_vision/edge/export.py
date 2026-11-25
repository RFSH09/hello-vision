"""Edge export plans: the commands that turn an ONNX model into each runtime's format.

These only BUILD command lines (and an RKNN script); nothing here has been executed against a
real model or on real boards. Run them on the matching device or toolchain host.
"""

from __future__ import annotations

from pathlib import Path

TARGETS = ("onnx-cpu", "tensorrt", "openvino", "rknn")


def plan_export(
    onnx_path: Path, target: str, out_dir: Path, fp16: bool = True, rknn_platform: str = "rk3588"
) -> list[str]:
    """Return the shell command (as argv) that converts ``onnx_path`` for ``target``.

    Args:
        onnx_path: Source ONNX model.
        target: One of :data:`TARGETS`.
        out_dir: Directory for the converted artefact.
        fp16: Use half precision where the target supports it.
        rknn_platform: Rockchip chip name for RKNN.

    Raises:
        ValueError: Unknown target.
    """
    stem = onnx_path.stem
    if target == "onnx-cpu":
        return ["python", "-m", "onnxruntime.tools.convert_onnx_models_to_ort", str(onnx_path)]
    if target == "tensorrt":  # on the Jetson, with TensorRT installed
        cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={out_dir / (stem + '.engine')}"]
        return [*cmd, "--fp16"] if fp16 else cmd
    if target == "openvino":  # OpenVINO model converter
        cmd = ["ovc", str(onnx_path), "--output_model", str(out_dir / (stem + ".xml"))]
        return [*cmd, "--compress_to_fp16", str(fp16).lower()]
    if target == "rknn":  # run the script below on a PC with rknn-toolkit2
        return ["python", str(out_dir / "convert_rknn.py"), str(onnx_path), rknn_platform]
    raise ValueError(f"unknown target {target!r}; choose from {TARGETS}")


def rknn_script(quantize: bool = False) -> str:
    """Return a ``rknn-toolkit2`` conversion script (``convert_rknn.py``) as text."""
    return f'''"""Convert an ONNX model to RKNN (run on a PC with rknn-toolkit2 installed)."""
import sys
from rknn.api import RKNN

onnx_path, platform = sys.argv[1], sys.argv[2]
rknn = RKNN()
rknn.config(target_platform=platform)
assert rknn.load_onnx(model=onnx_path) == 0
assert rknn.build(do_quantization={quantize}) == 0
assert rknn.export_rknn(onnx_path.rsplit(".", 1)[0] + ".rknn") == 0
'''

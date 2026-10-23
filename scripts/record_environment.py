"""Record everything that can change a result between machines: tool versions, fonts, GPU, the
execution provider ONNX Runtime really uses, and a hash of the generated test images.

    python scripts/record_environment.py DATA_DIR MODEL.onnx OUT.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def java_version() -> str:
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=30)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "version" in ln and "warning" not in ln]
        return lines[0] if lines else "unavailable"
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def sh(cmd: list[str]) -> str:
    try:
        return (lambda r: (r.stdout + r.stderr).strip().splitlines()[0])(subprocess.run(cmd, capture_output=True, text=True, timeout=30))
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def main(data: str, model: str, out: str) -> dict:
    from findoc.synth import fonts_available

    imgs = sorted((Path(data) / "images" / "test").glob("*.jpg"))
    h = hashlib.sha256()
    for p in imgs:
        h.update(p.read_bytes())
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "tesseract": sh(["tesseract", "--version"]), "java": java_version(),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "packages": {p: ver(p) for p in ["ultralytics", "torch", "onnxruntime", "onnxruntime-gpu", "pyspark", "pytesseract", "opencv-python-headless", "numpy", "pillow"]},
        "fonts_found": fonts_available(),
        "test_images": len(imgs), "test_images_sha256": h.hexdigest(),
    }
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(model, providers=ort.get_available_providers())
        info["onnx_session_providers"] = sess.get_providers()
        info["onnx_available_providers"] = ort.get_available_providers()
        from ultralytics import YOLO
        m = YOLO(model, task="detect")
        m(np.zeros((640, 960, 3), np.uint8), device=0 if info["gpu"][:11] != "unavailable" else "cpu", verbose=False)
        be = getattr(getattr(m, "predictor", None), "model", None)
        sess2 = getattr(be, "session", None)
        info["ultralytics_session_providers"] = sess2.get_providers() if sess2 else "not exposed"
    except Exception as e:  # noqa: BLE001
        info["provider_check_error"] = f"{type(e).__name__}: {e}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main(*sys.argv[1:4])

#!/usr/bin/env python3
"""Production TRT engine build entrypoint.

This wrapper keeps the runtime contract stable at tools/build-trt-engine.py
while reusing the existing implementation in tools/onnx-export/build-trt-engine.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve().parent / "onnx-export" / "build-trt-engine.py"
    onnx_export_dir = str(script_path.parent)
    if onnx_export_dir not in sys.path:
        sys.path.insert(0, onnx_export_dir)
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

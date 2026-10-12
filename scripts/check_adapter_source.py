"""Type-check the original adapter against an explicit external BMI header.

No header download, linking, shared-library loading, CFE/VM call, or fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


BMI_SHA256 = "ef287f3c3d17b887561fb0dbfcd2cd926fb61510e3e5e9bb38bede367a5bb595"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmi-include", required=True, type=Path)
    args = parser.parse_args()
    header = args.bmi_include / "bmi.h"
    if header.is_symlink() or not header.is_file() or header.stat().st_size > 32768:
        raise SystemExit("Explicit regular small external BMI header required")
    if hashlib.sha256(header.read_bytes()).hexdigest() != BMI_SHA256:
        raise SystemExit("External BMI header differs from the recorded source interface")
    compiler = shutil.which("cc")
    if not compiler:
        raise SystemExit("A local C11 compiler is required")
    root = Path(__file__).resolve().parents[1]
    argv = [compiler, "-std=c11", "-fsyntax-only", "-I", str(args.bmi_include),
            "-I", str(root / "native"), str(root / "native/cfe_adapter/neural_bridge.c")]
    positive = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if positive.returncode:
        raise SystemExit("Adapter source type-check failed")
    unsafe = subprocess.run(argv[:2] + ["-ffast-math"] + argv[2:], capture_output=True, text=True, timeout=30)
    if unsafe.returncode == 0 or "Neural bridge validation requires ordinary finite IEEE math" not in unsafe.stderr:
        raise SystemExit("Adapter IEEE/finite-math rejection guard was not preserved")
    print(json.dumps({"passed": True, "scope": "source syntax and unsafe-math rejection only",
                      "external_BMI_header_sha256": BMI_SHA256, "compile_checks": 2,
                      "linked_or_loaded_native_libraries": 0, "hydrologic_model_calls": 0,
                      "third_party_header_or_implementation_redistributed": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Compile the standalone original C kernels and run their generated fixtures.

No NextGen/BMI code is linked. Every build lives in an automatically removed
temporary directory. Each subprocess has a 30-second wall timeout.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which("cc")
    if compiler is None:
        raise SystemExit("A C99 compiler (cc) is required for this optional check.")
    if sys.platform == "win32":
        raise SystemExit("This build helper supports macOS/Linux compilers only.")
    with tempfile.TemporaryDirectory(prefix="hydra-kernels-") as folder:
        library = Path(folder) / ("kernels.dylib" if sys.platform == "darwin" else "kernels.so")
        command = [compiler, "-std=c99", "-O2", "-ffp-contract=off", "-fPIC",
                   "-dynamiclib" if sys.platform == "darwin" else "-shared",
                   str(root / "native/kernels.c"), "-o", str(library), "-lm"]
        subprocess.run(command, check=True, timeout=30)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   HYDRA_KERNEL_LIBRARY=str(library),
                   HYDRA_KERNEL_LIBRARY_SHA256=hashlib.sha256(library.read_bytes()).hexdigest(),
                   HYDRA_KERNEL_REPORT=str(Path(folder) / "report.json"))
        subprocess.run([sys.executable, "-B", str(root / "tests/test_native_kernels.py")],
                       env=env, check=True, timeout=30)
        print((Path(folder) / "report.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

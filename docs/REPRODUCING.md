# Reproducing the public checks

The core supports Python 3.10+ and has no runtime dependencies. Install from a fresh checkout with `python -m pip install .` in a virtual environment. Packaging uses setuptools. No credentials, datasets, native hydrology library, GPU or service is required.

Run `hydra-correctors` or `python -m hydra_correctors`. Both print the same deterministic JSON shown in `examples/demo-output.json`. Compare numeric values with normal floating-point tolerance across platforms; cross-platform libm bitwise equality is not claimed.

Run the evaluation and demonstration tests explicitly:

```sh
python -m unittest discover -s tests -p test_evaluation.py -v
python -m unittest discover -s tests -p test_demo.py -v
python scripts/check_native.py
```

The native helper requires `cc` and supports macOS/Linux. It compiles only `native/kernels.c`, uses no downloaded source, allows 30 seconds per subprocess and deletes its temporary build. The 15 native fixture methods cap successful forward calls at 4,096. Prospective C/Python agreement tolerances are absolute `2e-14`, relative `2e-13`. Do not use unrestricted test discovery: `test_native_kernels.py` expects the helper's freshly built library and content hash.

## Optional outlet model

The retained outlet model uses the source environment's exact package versions. MLX platform support is narrower than the dependency-free core; the locally checked environment is Apple silicon with Python 3.12. This optional environment is not required for the default demo.

```sh
python -m pip install '.[outlet]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/check_outlet.py
```

The check explicitly selects CPU, creates scratch parameters, switches to evaluation mode and performs two forward calls for one `[1,168,10]` synthetic input. It checks finite `[1,18]` residual output, exact same-process replay and float64 rejection. It does not fit, load a checkpoint, score hydrologic skill, run a Mamba test or prove parity with the original thesis model.

The bare outlet classes are research numerical components. Callers must independently enforce feature provenance, masks, valid site indices, scaling and availability; passing shapes alone is insufficient.

See [VALIDATION.md](VALIDATION.md) for the exact checks actually completed for this publication.

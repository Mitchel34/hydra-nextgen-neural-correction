# Reproducing the public checks

The core supports Python 3.10+ and has no runtime dependencies. Install from a fresh checkout with `python -m pip install .` in a virtual environment. Packaging uses setuptools. No credentials, datasets, native hydrology library, GPU or service is required.

Run `hydra-correctors` or `python -m hydra_correctors`. Both print the same deterministic JSON shown in `examples/demo-output.json`. Compare numeric values with normal floating-point tolerance across platforms; cross-platform libm bitwise equality is not claimed.

Run the evaluation and demonstration tests explicitly:

```sh
python -m unittest discover -s tests -p test_evaluation.py -v
python -m unittest discover -s tests -p test_demo.py -v
python scripts/check_native.py
```

The native helper selects this checkout’s Python source explicitly and requires `cc` and supports macOS/Linux. It compiles only `native/kernels.c`, uses no downloaded source, allows 30 seconds per subprocess and deletes its temporary build. The 15 native fixture methods cap successful forward calls at 4,096. Prospective C/Python agreement tolerances are absolute `2e-14`, relative `2e-13`. Do not use unrestricted test discovery: `test_native_kernels.py` expects the helper's freshly built library and content hash.

## Portable artifact and synthetic state checks

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_bridge_artifacts.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_process_boundary.py -v
python scripts/verify_source.py
```

All fixtures and model coefficients are generated in memory or temporary directories. The first group verifies the original canonical artifact codec; the second uses an invented bucket and sentinel. No physical hydrology, fitted model, private source or campaign authority is accessed. See [BRIDGE.md](BRIDGE.md).

To type-check the original C/BMI adapter, obtain the exact externally maintained header described in [its dependency manifest](../native/cfe_adapter/DEPENDENCIES.json), then run:

```sh
python scripts/check_adapter_source.py --bmi-include /path/to/external/include
```

The helper verifies the header hash and runs two C11 syntax-only compiler calls, including an expected unsafe-math rejection. It neither downloads a header nor links, loads or executes CFE. No native runtime configuration is supplied.

## Optional outlet model

The retained outlet model uses the source environment's exact package versions. MLX platform support is narrower than the dependency-free core; the locally checked environment is Apple silicon with Python 3.12. This optional environment is not required for the default demo.

```sh
python -m pip install '.[outlet]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/check_outlet.py
```

The check explicitly selects CPU, creates scratch parameters, switches to evaluation mode and performs two forward calls for one `[1,168,10]` synthetic input. It checks finite `[1,18]` residual output, exact same-process replay and float64 rejection. It does not fit, load a checkpoint, score hydrologic skill, run a Mamba test or prove parity with the original thesis model.

The bare outlet classes are research numerical components. Callers must independently enforce feature provenance, masks, valid site indices, scaling and availability; passing shapes alone is insufficient.

See [VALIDATION.md](VALIDATION.md) for the exact checks actually completed for this publication.

## Retained real-data baseline aggregates

Run `python scripts/check_baseline.py` and `PYTHONPATH=src python -m unittest discover -s tests -p test_baseline_report.py -v`. These use the checked-in [C evaluation bundle](../results/c_physical_baseline_2019_10/README.md), reproduce sufficient-statistic arithmetic and check hashes/scope. They do not load raw forecasts, observations or model weights. For the optional figure, install the `report` extra in an isolated environment and run `python scripts/plot_baseline.py`. Matplotlib is not required by the core demo or baseline arithmetic.

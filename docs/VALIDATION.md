# Publication validation — 7 October 2026

These are engineering checks on the curated public files. They establish no hydrologic forecast skill.

| Check actually executed | Result |
| --- | --- |
| Fresh Python 3.14 virtual environment, wheel built from the public package and installed with `--no-index --no-deps` | Pass; no runtime dependency downloads; `pip check` reports no broken requirements |
| Installed CLI versus checked-in synthetic example | Exact JSON equality on this host |
| Evaluation guards | 17 tests passed |
| Demo replay/declared-synthetic checks | 1 test passed |
| Standalone C99/Python kernel agreement and rejection tests | 15 tests passed; 89 successful and 43 rejected C calls; 89 successful and 10 rejected Python calls |
| Optional retained GRU–Transformer V3 | Pass on Apple silicon, Python 3.12, MLX 0.32.2, NumPy 2.5.3; CPU, dense attention, 219,049 parameters, two forward calls, finite `[1,18]` output, exact same-process replay, float64 rejection |

The clean installed core test groups completed in approximately 0.07, 0.09 and 0.63 seconds respectively, including C compilation for the last group. The optional outlet subprocess completed in about 1.85 seconds with a 10-second CPU limit and 25-second wall timeout. These timings describe this small check on one host; they are not model-training or deployment benchmarks.

The first optional MLX attempt failed during device initialization in the restricted sandbox, before model construction. The isolated host-level retry passed with CPU selected explicitly. No permissions or security settings were changed. MLX may require host device initialization even for CPU inference; the dependency-free demo does not have this requirement.

All inputs and coefficients used here are synthetic or analytic. Zero scientific fits, hydrologic model runs, real-data evaluations, archive rebuilds or paid jobs were performed. No heavy shared-host job was acquired or launched. Existing research state was read only.

The optional MLX check reused already installed dependencies. A fresh optional-dependency installation, other operating systems, Mamba inference/training, gradients, scientific data preparation, physical coupling, water/energy conservation and optimized-kernel equivalence were not validated by this release. The fresh-install result applies to the dependency-free core.

The source hashes and transformations are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Independent source-boundary and exact-tree privacy review records are retained with the publication checkpoint; private paths and operational records are deliberately excluded from the public tree.

## Consolidation validation — 12 October 2026

The existing 7 October validation above remains historical. The current public update adds a retained real-data physical-baseline aggregate bundle and original small report/check code. It does not rerun the physical experiment or read the raw forecasts/observations.

| Check actually executed | Result |
| --- | --- |
| Existing evaluation guards, synthetic demo, and new retained-baseline tests | 24 passed: 17 + 1 + 6; zero failures/errors |
| Original standalone native C/Python inference fixtures | 15 passed; 89 successful/43 rejected C calls, 89 successful/10 rejected Python calls |
| C aggregate evidence and count/scope/hash check | Pass; 13,203 cells and the saved pooled RMSE/MAE/bias reproduced |
| Existing extraction/third-party license provenance | All 11 published and current original-source hashes matched |
| Combined existing/new source index | 16 provenance-bound files matched |
| Quantitative figure | Rendered and visually checked; Matplotlib 3.11.2, NumPy 2.5.4, Python 3.12; no clipping or mislabeled comparison |

The dependency-free software groups completed in about 0.06 seconds; the native fixture helper (including C compilation) completed in about 0.99 seconds on this host. These are small software-check timings, not training or deployment benchmarks. The optional plotting dependencies were isolated for figure generation and did not change the research runtime. No scientific fits, native hydrologic model calls, archive regeneration or campaign restart occurred. The unchanged optional MLX path was not rerun.

The machine-readable [software evaluation](../results/software_checks_2026_10_12.json), [C report](../results/c_physical_baseline_2019_10/README.md), [metric provenance](../results/c_physical_baseline_2019_10/provenance.json) and [artifact checksums](../results/c_physical_baseline_2019_10/CHECKSUMS.sha256) are retained in Git. Real-data metric evidence is separate from generated kernel fixtures. Raw result records with private process/source-path details are excluded. Independent parent review of the frozen publication tree is still required before push.

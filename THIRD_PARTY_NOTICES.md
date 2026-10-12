# Third-party notices

The retained outlet source was extracted from the current corrected-error model implementation. Its original notice headers are preserved.

| Source portion | Upstream identity | Terms |
| --- | --- | --- |
| Mamba-2/SSD adaptation in `src/hydra_correctors/outlet/preserved_memory_mlx.py` | [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa), commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` | Copyright (c) 2024 Tri Dao, Albert Gu; [Apache License 2.0](src/hydra_correctors/outlet/licenses/MAMBA_APACHE_2_0.txt) |
| MLX Mamba-2 adaptation retained in that module | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/tree/dcbcf786c0cf56f9a12fabe9468c887781431ae2), commit `dcbcf786c0cf56f9a12fabe9468c887781431ae2` | Copyright (c) 2025 Apple Inc.; [MIT](src/hydra_correctors/outlet/licenses/MLX_LM_MIT.txt) |

The adaptations implement local MLX sequence execution and residual integration; they are not presented as unchanged upstream files. This publication copies the three local outlet modules without further mathematical changes. The Mamba path is preserved as part of the shared model module but is not exercised by this release's optional GRU check. Exact local-source and publication hashes are in [the provenance manifest](docs/SOURCE_PROVENANCE.json).

MLX and NumPy are optional installed dependencies, not vendored packages. The C/Python internal kernels and evaluation guards are original local HYDRA research source. The original HYDRA C/BMI adapter is included as a caller of an external interface. No upstream NextGen, CFE, Noah, BMI header or t-route implementation is included. The exact external BMI header carries the CSDMS MIT notice, while the recorded external CFE checkout carries Apache-2.0; links and hashes are recorded in [the adapter dependency manifest](native/cfe_adapter/DEPENDENCIES.json). These dependencies are not redistributed. References to those projects describe the research context, not a redistribution or license grant.

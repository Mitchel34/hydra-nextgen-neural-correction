# Working and publication map

**This public repository is the authoritative development and publication home for the portable HYDRA correction kernels, evaluation contracts, aggregate-report code, curated result tables/figures and accompanying documentation.** The current checkout is named `HYDRA-NextGen-Neural-Correction`; its origin is [Mitchel34/hydra-nextgen-neural-correction](https://github.com/Mitchel34/hydra-nextgen-neural-correction).

| Layer | Maintained here | Retained externally |
| --- | --- | --- |
| Portable correction/evaluation software | `native/`, `src/hydra_correctors/`, bounded public checks/tests | Historical source copies remain evidence snapshots, not a second evolving portable package |
| Scientific findings | Small metrics CSV/JSON, reports, figures, disabled evaluation manifests, source/checksum indices under `results/` | Forecasts, observations/weather, weights, full data joins and large archives |
| Native research execution | Interface/design documentation only | Existing qualified runtime, physical-model components, scientific authority, resource registry and all original run receipts |
| Earlier research | Clear separate links and status boundaries | NWM thesis, separate statistical experiments and independently evolving AI-method candidates |

Future portable code changes start here, on reviewed branches. A research consumer must pin a public commit and qualify a prospective adapter before using changed code. Existing executed/private sources are frozen scientific history; they are neither silently overwritten nor relabeled as current Git code. Native operational controls remain authoritative in their existing private campaign. This publication does not relocate their data, modify ownership or start another runner.

## Scientific paths

**Original HYDRA** learns outlet streamflow residuals from issue-available history and forecast/error features. The earlier [NWM thesis](https://github.com/Mitchel34/hydra-nwm-streamflow-correction) is distinct from the retained NextGen outlet architecture here. Earlier fitted studies are not revalidated or compared in this update.

**Internal Noah→CFE correction** investigates a bounded demand/process-interface factor before CFE's physical update. The public original MLP/GRU kernels provide candidate untrained factors/states; this package does not implement or validate the complete native coupled transaction. The [October 2019 C result](../results/c_physical_baseline_2019_10/README.md) is a fixed physical reference, with zero learned fits and no paired corrected comparator.

Generation software and [historical acquisition tooling](https://github.com/Mitchel34/NextGen_Hydra) have separate responsibilities. A portal or data acquisition package is not a replacement for the scientific research code.

## Pinned project and local configuration

Keep the existing pinned `ngen reforecast` project. Attach the existing public checkout using **Edit project → Add folder**, then **Make primary** if this is the selected development root; keep the old organizing folder as secondary. No data move is required. New-chat cwd, Git/PR/worktree actions and automatic AGENTS/skills/config discovery use the primary folder. See [official project documentation](https://learn.chatgpt.com/docs/projects). Existing task cwd and scientific ownership remain separate from sidebar placement.

The earlier private-checkout recommendation is superseded by the owner's selected public repository. The uninstalled private documentation draft remains retained locally as planning history. This change does not activate any model setting: [project-context.proposed.toml](../configs/project-context.proposed.toml) is a separate inactive proposal, preserving context `872000`, compaction `697600` and scope `total`. It can be considered only after primary-folder selection. The global config and existing cloud thread's effective settings are not changed or inferred.

Start with the [publication runbook](RUNBOOK.md), [status](STATUS.md), [reproduction instructions](REPRODUCING.md) and [public result bundle](../results/c_physical_baseline_2019_10/README.md).

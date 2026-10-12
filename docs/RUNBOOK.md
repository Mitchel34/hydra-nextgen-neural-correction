# Publication and evaluation runbook

1. Develop portable source and curated scientific findings in this repository. Keep original private execution trees, data and failed-run history unchanged.
2. Run the dependency-free demo/evaluation checks, retained baseline evidence check and the small native-kernel helper. These are software/aggregate checks, not training or reforecast generation.
3. For a new real-data finding, record exact issue window, lead/support counts, units, baseline versus corrected roles, fitted versus fixed status, exclusions and remaining scientific limitations. Never rank different A/B/C windows as a paired improvement result.
4. Commit small results/evaluation CSV/JSON, tables, reports, figures, code/configuration and provenance. Keep bulk forecasts/targets, weights, private operational records and credentials outside Git. Do not apply a blanket `results/` ignore.
5. Review an explicit path allowlist, full staged diff, scientific evidence, source rights and commit history. Obtain the requested independent parent review of the frozen tree/commit before the ordinary publication push. Any changed bytes require review of the changed tree.
6. After review passes, push the reviewed branch normally and inspect the public commit/README/result links. Preserve main history and the existing license status; no force push or private-repository publication is allowed.

## Available safe entrypoints

```sh
PYTHONPATH=src python -m hydra_correctors
PYTHONPATH=src python -m unittest discover -s tests -p test_evaluation.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_demo.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_baseline_report.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_bridge_artifacts.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_process_boundary.py -v
python scripts/check_baseline.py
python scripts/check_native.py
python scripts/verify_source.py
```

The baseline checker reads only checked-in sufficient statistics and named small metadata files; it cannot access external forecasts or initiate a physical run. [The report manifest](../results/c_physical_baseline_2019_10/run_manifest.json) is explicitly disabled. Optional Matplotlib figure generation reads the same aggregates; the MLX outlet check remains optional and synthetic.

The [portable bridge checks](../results/bridge_checks_2026_10_12.json) and [source provenance](BRIDGE_SOURCE_PROVENANCE.json) record generated-fixture and adapter-source results. The optional external-header syntax checker is described in [BRIDGE.md](BRIDGE.md). It carries no permission to launch a native campaign.

The current public checkpoint is the [C baseline summary](../results/c_physical_baseline_2019_10/summary.json), its [scope manifest](../results/c_physical_baseline_2019_10/run_manifest.json), [provenance](../results/c_physical_baseline_2019_10/provenance.json) and [status](STATUS.md). They record findings, not private scientific execution authority or resource allocations.

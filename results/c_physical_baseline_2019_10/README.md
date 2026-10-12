# C physical baseline — October 2019

This is a saved **physical reference evaluation**, not a neural-correction result. Forecast issues run hourly from **2019-10-01 00:00 UTC inclusive to 2019-11-01 00:00 UTC exclusive**, with 1–18-hour leads at USGS **03479000**, parameter **00060**, outlet segment **1017693**. The partition is **C outlet-fit development**. It is a constructed idealized-state research reforecast, not archived operational NextGen guidance.

| Retained evaluation | Value |
| --- | ---: |
| Issued forecasts | 744 |
| Raw forecast cells | 13,392 |
| Scored cells | 13,203 |
| Excluded cells under frozen support/maturity rules | 189 |
| Unique C observation hours | 742 |
| Pooled RMSE | 11.686679365777943 m³/s |
| Pooled MAE | 6.59026054242611 m³/s |
| Mean forecast − observation bias | +5.972715863770979 m³/s |
| Learned fits in this comparison | 0 |

![RMSE, MAE and signed bias by forecast lead for the physical baseline](lead_metrics.svg)

Read [metrics_by_lead.csv](metrics_by_lead.csv), [summary.json](summary.json), the [disabled evaluation manifest](run_manifest.json) and [provenance](provenance.json). Small metrics, reports, evaluation configuration, code and figures belong in Git. Forecast/target arrays, weights, private campaign records and credentials do not. The manifest identifies the externally retained forecast artifact by non-sensitive ID, length and checksum, without a local path or public data-access claim.

## Reproduce the published arithmetic

From the repository root:

```sh
python scripts/check_baseline.py
PYTHONPATH=src python -m unittest discover -s tests -p test_baseline_report.py -v
```

These commands read only the checked-in aggregate report. They recompute each lead's RMSE/MAE/bias and pooled errors from sufficient statistics, validate support counts, hash-check the published artifacts and reject unsupported learned/paired-comparison claims. No hydrologic model, observation reader, fit or network call is used. Pooling uses error totals weighted by scored-cell counts; the arithmetic mean of per-lead RMSE is a different statistic.

The retained `normalized_RMSE` column uses the already-fixed source reference scale **27.25475208263832 m³/s**; the scale was not fitted on C. It is included for traceability, not as an A/B/C cross-window skill comparison.

Regenerate the figure with optional Matplotlib using `python scripts/plot_baseline.py`. The figure shows lead-specific support (742 scored cells at lead 1, decreasing to 725 at lead 18); it is not an uncertainty interval or an independent-sample assessment.

## Limits on interpretation

- Observation availability uses a retrospective valid-time-plus-one-hour assumption. Real publication latency was not authenticated.
- C is development data. No untouched holdout or paired corrected result is supplied.
- Earlier A/B/outlet studies have different windows/supports; this report does not rank them or claim improvement.
- Original C8/C9 native outcomes remain UNKNOWN. Subsequent independent numerical recovery, composition and evaluation do not recast those whole operations as PASS.
- Full water/energy closure, hidden land-state equality, kernel-native branch parenthood and portable restart remain unproven.
- Reproducing these aggregate errors does not regenerate the physical forecasts or authenticate a raw observation vintage. Raw inputs remain external.
- The internal MLP/GRU kernels are untrained inference components; the retained outlet architecture is source without fitted weights. Neither supplies a corrected comparator for this report.

The original HYDRA outlet residual objective and internal Noah→CFE process-interface correction are distinct research paths. This baseline belongs to the latter's physical reference preparation. It establishes no learned correction benefit.

# Comparing forecasts on common support

`hydra_correctors.paired_comparison.compare_paired` scores a reference and a
candidate in one call using the same registered forecast keys, observations,
missing-target masks and gauge scales. It reuses `evaluation.score_paired` and
requires a finite forecast from both models for every registered key, including
keys with missing observations. Missing or extra predictions fail the comparison;
there is no automatic intersection of successful forecast rows.

This is evaluation preparation for comparisons such as internal correction
versus an outlet postprocessor. The public check uses invented values only. It
does not supply a paired learned-correction result, fitted weights, physical
forecasts or a training runner.

## Minimal synthetic example

```python
from hydra_correctors.evaluation import ForecastKey
from hydra_correctors.paired_comparison import compare_paired

key = ForecastKey("synthetic-gauge", issue_epoch=0, lead=1)
comparison = compare_paired(
    [key], observations={key: 10.0},
    baseline_predictions={key: 12.0}, candidate_predictions={key: 11.0},
    gauge_scales={"synthetic-gauge": 2.0},
    baseline_id="invented-reference", candidate_id="invented-candidate",
)
assert comparison["baseline"]["primary"] == 1.0
assert comparison["candidate"]["primary"] == 0.5
assert comparison["primary_delta"] == -0.5
```

The result contains both complete score dictionaries, `primary_delta`, and
RMSE/MAE/normalized-RMSE differences for each gauge/lead cell. All differences
are **candidate minus baseline**, so negative values mean lower error. The
primary endpoint averages normalized RMSE equally over leads within each gauge,
then equally over gauges. Its difference retains that weighting; it is not a
pooled RMSE difference or necessarily an equally weighted mean of all cell
differences. Zero baseline error is valid because no percentage improvement is
computed. PBIAS remains a signed diagnostic in each score; zero observation sums
retain the scorer's undefined-bias reason.

## Reproducible comparison context

`context_sha256` hashes a canonical JSON object with schema
`hydra.paired-evaluation-context.v1`, sorted forecast keys and observed values
(including `None` masks), and sorted gauge scales. Observations and scales are
normalized to floats. Reordering keys or mappings does not change this digest;
changing a key, target, mask or scale does. Predictions and caller IDs are
excluded, so exchanging the two forecasts preserves context and reverses the
error differences.

The digest records the supplied context. It does not authenticate source data,
prediction contents, model identity, historical observation availability or
model lineage. The two required distinct IDs are caller labels. The function
cannot determine whether support was registered before results were inspected,
whether shared scales came from fitting data, or whether forecast features were
causal. Callers must establish those facts and compatible discharge units,
partition roles, fit/selection boundaries and physical qualification separately.
Support from different assessment windows must not be presented as paired.

These arithmetic differences provide no uncertainty interval or significance
test. Overlapping issue trajectories do not become independent samples, and
their sums are not an event-volume integral. No skill or training qualification
follows from passing the generated checks.

## Run the focused checks

From the source checkout, using Python 3.10+ with no runtime dependencies:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_paired_comparison.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_evaluation.py -v
```

The 14 new generated test methods cover shared aggregation, sign reversal,
context identity, finite complete forecasts on both sides, zero reference
error, masks, invalid inputs and preservation of caller inputs. Source and test
hashes are recorded in [the provenance index](CONSOLIDATION_PROVENANCE.json).

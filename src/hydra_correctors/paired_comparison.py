"""Two-forecast comparisons on one declared evaluation context.

This module reuses the existing scorer. It authenticates neither observations
nor model lineage, and performs no fitting or scientific forecast execution.
"""
from collections.abc import Mapping
import hashlib
import json

from .evaluation import score_paired


CONTEXT_SCHEMA = "hydra.paired-evaluation-context.v1"


def compare_paired(expected_keys, observations, baseline_predictions,
                   candidate_predictions, gauge_scales, *, baseline_id,
                   candidate_id):
    """Score two complete forecast mappings with shared keys, targets and scales.

    IDs are distinct, nonblank caller labels, not verified model identities.
    Error deltas are candidate minus baseline; negative values mean lower error.
    The primary delta retains score_paired's equal-gauge/equal-lead weighting.
    PBIAS remains a signed diagnostic in each score and is not differenced.

    The versioned context digest binds sorted forecast keys, observed values
    (including None masks), and common scales. Numeric values are normalized to
    floats, matching the scorer. Predictions and IDs are excluded: this digest
    identifies declared comparison inputs, not source or model authenticity.
    """
    for name, value in (("baseline_id", baseline_id), ("candidate_id", candidate_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonblank identity")
    if baseline_id == candidate_id:
        raise ValueError("baseline and candidate identities must differ")

    # Freeze the shared context before either scoring call. Requiring mappings
    # also avoids silently collapsing duplicate keys in a sequence of pairs.
    for name, values in (("observations", observations),
                         ("baseline_predictions", baseline_predictions),
                         ("candidate_predictions", candidate_predictions),
                         ("gauge_scales", gauge_scales)):
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping")
    keys = tuple(expected_keys)
    observations = dict(observations)
    gauge_scales = dict(gauge_scales)
    baseline_predictions = dict(baseline_predictions)
    candidate_predictions = dict(candidate_predictions)

    baseline = score_paired(keys, observations, baseline_predictions, gauge_scales)
    candidate = score_paired(keys, observations, candidate_predictions, gauge_scales)
    # Only validated keys and finite numeric inputs reach canonical serialization.
    context = {
        "schema": CONTEXT_SCHEMA,
        "targets": [
            [key.gauge, key.issue_epoch, key.lead,
             None if observations[key] is None else float(observations[key])]
            for key in sorted(keys, key=lambda key: (key.gauge, key.issue_epoch, key.lead))
        ],
        "gauge_scales": [[gauge, float(gauge_scales[gauge])]
                         for gauge in sorted(gauge_scales)],
    }
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cell_deltas = []
    for before, after in zip(baseline["cells"], candidate["cells"], strict=True):
        cell_deltas.append({
            "gauge": before["gauge"], "lead": before["lead"], "count": before["count"],
            **{f"{metric}_delta": after[metric] - before[metric]
               for metric in ("rmse", "mae", "nrmse")},
        })
    return {
        "status": "COMPARED_FIXED_SUPPORT",
        "baseline_id": baseline_id, "candidate_id": candidate_id,
        "context_schema": CONTEXT_SCHEMA,
        "context_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "delta_convention": "candidate_minus_baseline",
        "primary_delta": candidate["primary"] - baseline["primary"],
        "cell_deltas": cell_deltas,
        "baseline": baseline, "candidate": candidate,
    }

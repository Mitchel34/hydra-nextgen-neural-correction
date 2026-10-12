"""Validate published aggregate evidence without loading forecasts or targets.

Original HYDRA reporting code. Reproduces retained metric arithmetic only;
it cannot establish physical custody, data vintages, or learned improvement.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


FIELDS = (
    "lead_hours", "raw_forecast_cells", "scored_cells", "excluded_cells",
    "SSE_cms_squared", "SAE_cms", "signed_error_sum_cms", "RMSE_cms",
    "MAE_cms", "bias_forecast_minus_observation_cms", "normalized_RMSE",
)
COUNT_FIELDS = FIELDS[:4]
PUBLIC_FILES = ("metrics_by_lead.csv", "summary.json", "run_manifest.json")


def _close(actual: float, expected: float, name: str) -> None:
    if not math.isfinite(expected) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-10
    ):
        raise ValueError(f"Inconsistent {name}")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path, ceiling: int = 131072) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular local report file is required")
    with path.open("rb") as stream:
        raw = stream.read(ceiling + 1)
    if len(raw) > ceiling:
        raise ValueError("Report exceeds the small aggregate-file limit")
    return raw


def _json(path: Path) -> dict:
    result = json.loads(
        _read(path), object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON value: {value}")
        ),
    )
    if not isinstance(result, dict):
        raise ValueError("Report metadata must be an object")
    return result


def read_metrics(path: Path) -> list[dict]:
    reader = csv.DictReader(_read(Path(path)).decode("utf-8").splitlines())
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("Unexpected aggregate CSV fields")
    rows = []
    for original in reader:
        if set(original) != set(FIELDS) or any(v is None for v in original.values()):
            raise ValueError("Malformed aggregate row")
        try:
            row = {
                key: int(value) if key in COUNT_FIELDS else float(value)
                for key, value in original.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid numeric aggregate") from exc
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite aggregate")
        rows.append(row)
    if not rows:
        raise ValueError("No lead-level evidence")
    return rows


def validate_metrics(rows: list[dict], summary: dict) -> dict:
    """Check counts, per-lead statistics, and pooled sufficient statistics."""
    if summary.get("baseline_kind") != "physical_reference":
        raise ValueError("This bundle describes a physical reference only")
    if summary.get("learned_fits") != 0 or summary.get("learned_improvement_claim") is not False:
        raise ValueError("A baseline cannot be promoted to learned improvement")
    if summary.get("paired_corrected_comparison_available") is not False:
        raise ValueError("No paired corrected result is supplied by this bundle")
    window = summary["issue_window_UTC"]
    start = datetime.fromisoformat(window["start_inclusive"].replace("Z", "+00:00"))
    stop = datetime.fromisoformat(window["stop_exclusive"].replace("Z", "+00:00"))
    if start.utcoffset() is None or stop.utcoffset() is None:
        raise ValueError("Issue window must be timezone-aware")
    seconds = (stop - start).total_seconds()
    if seconds <= 0 or seconds % 3600:
        raise ValueError("Issue window must contain positive whole hours")
    issues = int(seconds // 3600)
    if summary["forecast_issues"] != issues:
        raise ValueError("Issue count disagrees with the declared hourly window")
    leads = summary["lead_hours"]
    if leads != list(range(1, 19)) or [row["lead_hours"] for row in rows] != leads:
        raise ValueError("Exactly one ordered row per declared 1–18-hour lead is required")
    scale = summary["normalization"]["fixed_reference_scale_cms"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be positive and finite")
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Unexpected metric fields")
        for key in COUNT_FIELDS:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("Counts must be nonnegative integers")
        for key in FIELDS[4:]:
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                raise ValueError("Statistics must be finite numbers")
        raw, count, excluded = (row[k] for k in COUNT_FIELDS[1:])
        if raw != issues or not 0 < count <= raw or count + excluded != raw:
            raise ValueError("Lead support does not reconcile")
        sse, sae, signed = (row[k] for k in FIELDS[4:7])
        if sse < 0 or sae < 0 or abs(signed) > sae + 1e-10:
            raise ValueError("Invalid error sufficient statistics")
        _close(math.sqrt(sse / count), row["RMSE_cms"], "lead RMSE")
        _close(sae / count, row["MAE_cms"], "lead MAE")
        _close(signed / count, row["bias_forecast_minus_observation_cms"], "lead bias")
        _close(row["RMSE_cms"] / scale, row["normalized_RMSE"], "fixed-scale RMSE")
    count = sum(row["scored_cells"] for row in rows)
    pooled = {
        "cells": count,
        "RMSE_cms": math.sqrt(math.fsum(row["SSE_cms_squared"] for row in rows) / count),
        "MAE_cms": math.fsum(row["SAE_cms"] for row in rows) / count,
        "bias_forecast_minus_observation_cms": math.fsum(row["signed_error_sum_cms"] for row in rows) / count,
    }
    if count != summary["scored_cells"] or count != summary["pooled"]["cells"]:
        raise ValueError("Pooled support disagrees")
    if sum(row["raw_forecast_cells"] for row in rows) != summary["raw_cells"]:
        raise ValueError("Raw support disagrees")
    if sum(row["excluded_cells"] for row in rows) != summary["excluded_cells"]:
        raise ValueError("Excluded support disagrees")
    for key in ("RMSE_cms", "MAE_cms", "bias_forecast_minus_observation_cms"):
        _close(pooled[key], summary["pooled"][key], "pooled " + key)
    return pooled


def validate_bundle(directory: Path) -> dict:
    """Read only the three named small public artifacts and their hash index."""
    directory = Path(directory)
    provenance = _json(directory / "provenance.json")
    pins = provenance["public_file_sha256"]
    if set(pins) != set(PUBLIC_FILES):
        raise ValueError("Unexpected public evidence paths")
    for name in PUBLIC_FILES:
        if hashlib.sha256(_read(directory / name)).hexdigest() != pins[name]:
            raise ValueError("Public artifact hash differs: " + name)
    summary = _json(directory / "summary.json")
    manifest = _json(directory / "run_manifest.json")
    if manifest.get("execution_enabled") is not False or manifest.get("physical_run_requested") is not False:
        raise ValueError("The published evaluation manifest must remain disabled")
    if manifest.get("gauge") != summary.get("gauge") or manifest.get("issue_window_UTC") != summary.get("issue_window_UTC"):
        raise ValueError("Manifest scope differs from the evaluated result")
    pooled = validate_metrics(read_metrics(directory / "metrics_by_lead.csv"), summary)
    return {"passed": True, "scope": "retained aggregate arithmetic and hashes only", "pooled_recomputed": pooled}

"""Dependency-free evaluation guards, not a scientific training/scoring runner.

Clock, mask and source-support contracts adapted from read-only AI-HYDRA
corrected_data.py and baselines.py; exact source hashes are bound by the fixture
runner. This module generalizes the original three-gauge/F1-F3 contracts.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from statistics import fmean, pstdev

HOUR = 3600


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _hour(value, name):
    value = _integer(value, name)
    if value % HOUR:
        raise ValueError(f"{name} must be an exact UTC hour")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _delay(value):
    value = _integer(value, "minimum observation delay")
    if value < 1:
        raise ValueError("minimum observation delay must be at least one hour")
    return value


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} identity is required")
    return value


@dataclass(frozen=True)
class ForecastKey:
    gauge: str
    issue_epoch: int
    lead: int

    def __post_init__(self):
        _identity(self.gauge, "gauge")
        object.__setattr__(self, "issue_epoch", _hour(self.issue_epoch, "issue"))
        lead = _integer(self.lead, "lead")
        if not 1 <= lead <= 18:
            raise ValueError("lead must be 1..18")
        object.__setattr__(self, "lead", lead)

    @property
    def valid_epoch(self):
        # Convert before multiplication; narrow integer leads cannot overflow.
        return self.issue_epoch + int(self.lead) * HOUR


@dataclass(frozen=True)
class Partition:
    name: str
    role: str
    start: int
    end: int
    valid_start: int | None = None
    label_available_by: int | None = None

    def __post_init__(self):
        _identity(self.name, "partition")
        if self.role not in ("fit", "select", "assess"):
            raise ValueError("unknown partition role")
        start, end = _hour(self.start, "start"), _hour(self.end, "end")
        valid_start = start if self.valid_start is None else _hour(self.valid_start, "valid_start")
        if not start <= valid_start < end:
            raise ValueError("invalid half-open partition")
        object.__setattr__(self, "valid_start", valid_start)
        if self.role in ("fit", "select") and self.label_available_by is None:
            raise ValueError("fit/select require label cutoff")
        if self.label_available_by is not None:
            cutoff = _hour(self.label_available_by, "label cutoff")
            if cutoff < end:
                raise ValueError("label cutoff precedes partition end")

    def accepts(self, key, available_epoch, minimum_observation_delay_hours=1):
        if not isinstance(key, ForecastKey):
            raise ValueError("ForecastKey required")
        available = _hour(available_epoch, "target availability")
        if available < key.valid_epoch:
            raise ValueError("target availability precedes valid time")
        available = max(available, key.valid_epoch + _delay(minimum_observation_delay_hours)*HOUR)
        return (self.start <= key.issue_epoch < self.end
                and self.valid_start <= key.valid_epoch < self.end
                and (self.label_available_by is None or available <= self.label_available_by))


@dataclass(frozen=True)
class Observation:
    gauge: str
    valid_epoch: int
    available_epoch: int
    value: float | None
    approved: bool
    source_id: str

    def __post_init__(self):
        _identity(self.gauge, "gauge")
        _identity(self.source_id, "observation source")
        _hour(self.valid_epoch, "valid time")
        _hour(self.available_epoch, "available time")
        if self.available_epoch < self.valid_epoch:
            raise ValueError("observation available before valid time")
        if type(self.approved) is not bool:
            raise ValueError("quality flag must be boolean")
        if self.value is not None:
            _finite(self.value, "observation")


def issue_available_value(observation, original_issue, minimum_observation_delay_hours=1):
    """Return approved finite value or a missing mask under a declared vintage.

    The source's availability may be a retrospective assumption. This function
    validates the declared clock; it cannot prove historical publication.
    """
    issue = _hour(original_issue, "original issue")
    if not isinstance(observation, Observation):
        raise ValueError("Observation required")
    available = max(observation.available_epoch, observation.valid_epoch + _delay(minimum_observation_delay_hours)*HOUR)
    if available > issue:
        raise ValueError("future observation/error is unavailable to issued forecast")
    present = observation.approved and observation.value is not None
    return {"value": observation.value if present else None, "mask": present,
            "age_hours": (issue-observation.valid_epoch)//HOUR,
            "source_id": observation.source_id, "effective_available_epoch": available}


@dataclass(frozen=True)
class Scale:
    gauge: str
    count: int
    mean: float
    std: float
    support_hash: str
    sources: tuple


def fit_unique_hour_scale(rows, partition, minimum_observation_delay_hours=1):
    """Synthetic-tested scale kernel; never called on real data by fixture runner."""
    if partition.role != "fit":
        raise ValueError("scaling requires fit support")
    delay = _delay(minimum_observation_delay_hours)
    rows = tuple(rows)
    if not rows:
        raise ValueError("empty scale support")
    keys, values, sources = set(), [], set()
    gauge = rows[0].gauge
    accepted = []
    for row in rows:
        if not isinstance(row, Observation) or row.gauge != gauge:
            raise ValueError("one declared gauge required")
        key = (row.gauge, row.valid_epoch)
        if key in keys:
            raise ValueError("duplicate unique-hour scale key")
        keys.add(key)
        if not partition.valid_start <= row.valid_epoch < partition.end:
            raise ValueError("scale row outside fit valid-time support")
        available = max(row.available_epoch, row.valid_epoch + delay*HOUR)
        if row.approved and row.value is not None and available <= partition.label_available_by:
            values.append(float(row.value)); sources.add(row.source_id)
            accepted.append((row.gauge, row.valid_epoch, available, row.source_id))
    if len(values) < 2:
        raise ValueError("insufficient scale support")
    try:
        std, mean = pstdev(values), fmean(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("nonfinite derived scale statistics") from exc
    if not math.isfinite(std) or not math.isfinite(mean) or std <= 0:
        raise ValueError("zero or invalid training scale")
    digest = hashlib.sha256(json.dumps(sorted(accepted),separators=(",", ":")).encode()).hexdigest()
    return Scale(gauge, len(values), mean, std, digest, tuple(sorted(sources)))


@dataclass(frozen=True)
class UpstreamLineage:
    model_id: str
    source_id: str
    configuration_id: str
    scaler_id: str
    replay_id: str
    fit_support: tuple
    selection_support: tuple
    scaler_support: tuple
    frozen_before_epoch: int
    frozen_weight_replay: bool

    def __post_init__(self):
        for field in ("model_id", "source_id", "configuration_id", "scaler_id", "replay_id"):
            _identity(getattr(self, field), field)
        _hour(self.frozen_before_epoch, "upstream freeze")
        if self.frozen_weight_replay is not True:
            raise ValueError("frozen-weight replay evidence required")
        if not self.fit_support or not self.scaler_support:
            raise ValueError("explicit upstream fit and scaler supports required")
        for collection, role in ((self.fit_support,"fit"),(self.selection_support,"select"),(self.scaler_support,"fit")):
            if not isinstance(collection, tuple):
                raise ValueError("immutable upstream support tuples required")
            for part in collection:
                if not isinstance(part, Partition) or part.role != role:
                    raise ValueError("upstream support role mismatch")
                if part.label_available_by > self.frozen_before_epoch:
                    raise ValueError("freeze predates upstream label availability")
        for fit in self.fit_support:
            for selection in self.selection_support:
                if fit.valid_start < selection.end and selection.valid_start < fit.end:
                    raise ValueError("upstream fit and selection target intervals overlap")
        for scaler in self.scaler_support:
            contained = any(fit.start <= scaler.start and fit.valid_start <= scaler.valid_start
                            and scaler.end <= fit.end
                            and scaler.label_available_by <= fit.label_available_by
                            for fit in self.fit_support)
            if not contained:
                raise ValueError("upstream scaler support must be contained in declared fit support")


@dataclass(frozen=True)
class StackedPrediction:
    key: ForecastKey
    value: float
    upstream: UpstreamLineage


def validate_disjoint_stack(predictions, expected_keys, outlet_fit):
    """Require all upstream fit, selection and transform labels before stack block.

    Metadata validation is necessary, not evidence that external native state
    replay actually occurred; receipt content hashes must be authenticated by
    the caller before constructing this lineage.
    """
    if outlet_fit.role != "fit":
        raise ValueError("outlet fit partition required")
    predictions, expected = tuple(predictions), tuple(expected_keys)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("nonempty unique expected keys required")
    actual = [p.key for p in predictions]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("stack predictions do not match frozen support")
    lineage = predictions[0].upstream
    if not isinstance(lineage, UpstreamLineage):
        raise ValueError("upstream lineage required")
    if lineage.frozen_before_epoch > outlet_fit.start:
        raise ValueError("upstream freeze reaches outlet fitting block")
    for part in lineage.fit_support+lineage.selection_support+lineage.scaler_support:
        if part.end > outlet_fit.start or part.label_available_by > outlet_fit.start:
            raise ValueError("upstream fit/selection/scaler labels overlap outlet fit")
    for pred in predictions:
        _finite(pred.value, "stack prediction")
        if pred.upstream != lineage:
            raise ValueError("mixed upstream model/configuration/scaler/replay identity")
        if not (outlet_fit.start <= pred.key.issue_epoch < outlet_fit.end
                and outlet_fit.valid_start <= pred.key.valid_epoch < outlet_fit.end):
            raise ValueError("stack target crosses outlet partition")
    return {"status":"DISJOINT_SUPPORT_VALIDATED", "rows":len(actual),
            "upstream_model_id":lineage.model_id, "replay_id":lineage.replay_id,
            "method":"disjoint_chronological_blocks", "native_replay_authenticated":False}


def score_paired(expected_keys, observations, predictions, gauge_scales):
    """Equal-gauge/equal-lead normalized RMSE on fixed support; no silent deletion.

    Every candidate must predict every registered key, including keys whose
    observation mask is missing. Missing labels do not become training targets.
    This computes no event-volume integral; PBIAS is a paired sum-of-samples
    bias diagnostic and overlapping issues are never described as a hydrograph.
    """
    keys = tuple(expected_keys)
    if not keys or any(not isinstance(k, ForecastKey) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("nonempty unique ForecastKeys required")
    frozen = set(keys)
    if set(observations) != frozen or set(predictions) != frozen:
        raise ValueError("observations/predictions must exactly match frozen keys")
    gauges = {k.gauge for k in keys}
    if set(gauge_scales) != gauges:
        raise ValueError("exact common gauge scales required")
    for value in gauge_scales.values():
        if _finite(value,"scale") <= 0:
            raise ValueError("positive common scale required")
    cells = {}
    for key in keys:
        predicted = _finite(predictions[key], "candidate prediction")
        cells.setdefault((key.gauge,key.lead), [])
        observed = observations[key]
        if observed is not None:
            observed = _finite(observed,"observation")
            cells[key.gauge,key.lead].append((observed,predicted))
    results, gauge_endpoints = [], {g:[] for g in gauges}
    for (gauge,lead), pairs in sorted(cells.items()):
        if not pairs:
            raise ValueError("registered gauge/lead has no scored target support")
        try:
            errors = [_finite(p-o,"derived error") for o,p in pairs]
            squares = [_finite(e*e,"squared error") for e in errors]
            rmse = _finite(math.sqrt(fmean(squares)),"RMSE")
            nrmse = _finite(rmse/gauge_scales[gauge],"normalized RMSE")
            mae = _finite(fmean(abs(e) for e in errors),"MAE")
            total = _finite(math.fsum(o for o,_ in pairs),"observation sum")
            bias = _finite(100*(math.fsum(errors)/total),"PBIAS") if total else None
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("nonfinite derived metric") from exc
        results.append({"gauge":gauge,"lead":lead,"count":len(pairs),"rmse":rmse,
                        "mae":mae,"nrmse":nrmse,"pbias_percent":bias,
                        "pbias_undefined":"zero_observation_sum" if total==0 else None})
        gauge_endpoints[gauge].append(nrmse)
    try:
        primary = _finite(fmean(fmean(v) for v in gauge_endpoints.values()),"primary endpoint")
    except OverflowError as exc:
        raise ValueError("nonfinite primary endpoint") from exc
    return {"status":"SCORED_FIXED_SUPPORT", "primary":primary,
            "cells":results,"registered_rows":len(keys),"scored_rows":sum(x['count'] for x in results),
            "missing_target_rows":sum(observations[k] is None for k in keys),
            "bias_convention":"positive_overprediction"}

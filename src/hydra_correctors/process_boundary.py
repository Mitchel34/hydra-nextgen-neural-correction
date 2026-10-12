"""Synthetic API/state fixture. No native model, neural network, or optimizer."""
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import math

HOUR = 3600
DEMAND = "water_potential_evaporation_flux"
LIQUID = "atmosphere_water__liquid_equivalent_precipitation_rate"
UNITS = {DEMAND: "m s-1", LIQUID: "mm h-1"}


class ContractError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ContractError(message)


def number(value, *, nonnegative=False):
    require(type(value) in (float, int) and math.isfinite(value), "finite numeric value required")
    require(not nonnegative or value >= 0, "negative value")
    return float(value)


def name(value):
    require(isinstance(value, str) and value and value.strip() == value and "\n" not in value, "explicit identity required")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ToyModel:
    policy_id: str
    parameters: tuple

    def __post_init__(self):
        name(self.policy_id)
        require(type(self.parameters) is tuple and 1 <= len(self.parameters) <= 16, "small immutable toy parameter tuple required")
        require(all(type(v) is float and math.isfinite(v) for v in self.parameters), "finite float toy parameters required")

    @property
    def model_id(self):
        return digest({"synthetic_policy": self.policy_id, "parameters": self.parameters})


@dataclass(frozen=True)
class Identity:
    catchment_id: str
    trajectory_id: str
    origin_epoch: int
    epoch: int
    model_id: str
    scaler_id: str
    input_id: str
    support_id: str

    def __post_init__(self):
        for key in (self.catchment_id, self.trajectory_id, self.model_id, self.scaler_id, self.input_id, self.support_id):
            name(key)
        require(type(self.origin_epoch) is int and type(self.epoch) is int and
                0 <= self.origin_epoch <= self.epoch and self.origin_epoch % HOUR == self.epoch % HOUR == 0,
                "hourly origin/current clock required")


@dataclass(frozen=True)
class Input:
    identity: Identity
    variable: str
    value: float
    units: str
    status: str = "SUCCESS"
    dtype: str = "double"
    itemsize: int = 8
    nbytes: int = 8


@dataclass(frozen=True)
class Sentinel:
    """Opaque ownership token, NOT a GRU hidden-state computation."""
    identity: Identity
    values: tuple

    def __post_init__(self):
        require(isinstance(self.identity, Identity) and type(self.values) is tuple and len(self.values) == 32,
                "immutable32 sentinel required")
        require(all(type(v) is float and math.isfinite(v) for v in self.values), "finite sentinel required")


@dataclass(frozen=True)
class FeatureFrame:
    identity: Identity
    raw_demand_m_s: float
    liquid_mm_h: float
    pre_update_soil_m: float

    @property
    def values(self):
        # Exactly d=3, in the declared source units. No fitted normalization.
        return (self.raw_demand_m_s, self.liquid_mm_h, self.pre_update_soil_m)


@dataclass(frozen=True)
class Readback:
    identity: Identity
    value: float
    status: str = "SUCCESS"
    units: str = "m"
    dtype: str = "double"
    itemsize: int = 8
    nbytes: int = 8


@dataclass(frozen=True)
class UpdateResult:
    identity: Identity
    soil_m: float
    status: str = "SUCCESS"


@dataclass(frozen=True)
class Decision:
    identity: Identity
    multiplier: float
    next_sentinel: Sentinel


def bounded_multiplier(raw):
    """The proposed bounded-head arithmetic only; not a neural head/model."""
    raw = number(raw)
    return math.exp(math.log(2.0) * math.tanh(raw))


class MockPhysical:
    """Simple arithmetic bucket for API tests; no CFE/Noah/routing equations."""
    def __init__(self, identity, soil_m):
        require(isinstance(identity, Identity) and identity.epoch == identity.origin_epoch, "fresh physical origin required")
        self._identity = identity
        self._soil_m = number(soil_m, nonnegative=True)
        self._claimant = None
        self._history = []

    def claim(self, claimant):
        require(self._claimant is None, "duplicate correction owner")
        self._claimant = claimant

    def snapshot(self):
        return {"identity": self._identity, "soil_m": self._soil_m, "history": copy.deepcopy(self._history)}

    def read_soil(self):
        return Readback(self._identity, self._soil_m)

    def update(self, claimant, identity, demand_m_s, liquid_mm_h):
        require(claimant is self._claimant and identity == self._identity, "physical owner/clock differs")
        demand = number(demand_m_s, nonnegative=True)
        rain = number(liquid_mm_h, nonnegative=True)
        available = self._soil_m + rain / 1000.0
        self._soil_m = available - min(available, demand * HOUR)
        self._history.append({"epoch": identity.epoch, "demand_m_s": demand, "liquid_mm_h": rain})
        self._identity = replace(identity, epoch=identity.epoch + HOUR)
        return UpdateResult(self._identity, self._soil_m)

    def _fork(self, identity, claimant):
        result = copy.deepcopy(self)
        result._identity = identity
        result._claimant = claimant
        return result


class Controller:
    def __init__(self, physical, model, scaler_id, policy):
        require(isinstance(model, ToyModel) and callable(policy), "toy model/policy required")
        name(scaler_id)
        snapshot = physical.snapshot()
        identity = snapshot["identity"]
        require(identity.epoch == identity.origin_epoch and not snapshot["history"], "changed model/scaler requires fresh origin replay")
        require(identity.model_id == model.model_id and identity.scaler_id == scaler_id, "model/scaler identity differs")
        self._identity, self._model, self._scaler_id = identity, model, scaler_id
        self._policy, self._physical = policy, physical
        self._claimant = object()
        physical.claim(self._claimant)
        self._sentinel = Sentinel(identity, (0.0,) * 32)
        self._inputs = {}
        self._phase, self._failure = "READY", None
        self._fork_ids = set()
        self._policy_calls, self._update_calls = 0, 0

    @property
    def identity(self): return self._identity

    @property
    def model(self): return self._model

    @property
    def sentinel(self): return self._sentinel

    def _live(self):
        require(self._phase == "READY", "trajectory is poisoned or busy; no reuse")
        require(self._model.model_id == self._identity.model_id and self._scaler_id == self._identity.scaler_id and
                self._sentinel.identity == self._identity, "model/scaler/state carryover differs; origin replay required")

    def _poison(self, error):
        self._phase, self._failure = "POISONED", type(error).__name__ + ": " + str(error)

    def set_input(self, packet):
        self._live()
        try:
            require(isinstance(packet, Input) and packet.identity == self._identity, "input owner/model/scaler/clock differs")
            require(packet.variable in UNITS and packet.variable not in self._inputs, "unknown or duplicate current input")
            require(packet.status == "SUCCESS" and packet.units == UNITS[packet.variable] and
                    packet.dtype == "double" and type(packet.itemsize) is int and packet.itemsize == 8 and
                    type(packet.nbytes) is int and packet.nbytes == 8, "input status/units/type/size differs")
            value = number(packet.value, nonnegative=True)
            self._inputs[packet.variable] = value
        except BaseException as error:
            self._poison(error)
            raise

    def frame(self):
        self._live()
        try:
            require(set(self._inputs) == set(UNITS), "both current inputs required before frame")
            value = self._physical.read_soil()
            require(isinstance(value, Readback) and value.identity == self._identity and value.status == "SUCCESS" and
                    value.units == "m" and value.dtype == "double" and type(value.itemsize) is int and
                    value.itemsize == 8 and type(value.nbytes) is int and value.nbytes == 8, "pre-update soil status/units/type/clock differs")
            soil = number(value.value, nonnegative=True)
            return FeatureFrame(self._identity, self._inputs[DEMAND], self._inputs[LIQUID], soil)
        except BaseException as error:
            self._poison(error)
            raise

    def step(self):
        self._live()
        try:
            frame = self.frame()
            before = self._physical.snapshot()
            require(before["identity"] == frame.identity and before["soil_m"] == frame.pre_update_soil_m,
                    "physical state changed during frame read")
            self._phase = "BUSY"
            self._policy_calls += 1
            decision = self._policy(frame, self._sentinel, self._model)
            require(self._model.model_id == frame.identity.model_id and self._scaler_id == frame.identity.scaler_id,
                    "model/scaler changed during decision")
            require(isinstance(decision, Decision) and decision.identity == frame.identity, "decision identity differs")
            factor = number(decision.multiplier)
            require(0.5 <= factor <= 2.0, "multiplier outside engineering bound")
            next_id = replace(frame.identity, epoch=frame.identity.epoch + HOUR)
            require(isinstance(decision.next_sentinel, Sentinel) and decision.next_sentinel.identity == next_id,
                    "sentinel owner/model/scaler/clock differs")
            require(self._physical.snapshot() == before, "policy mutated physical state")
            submitted = frame.raw_demand_m_s if factor == 1.0 else frame.raw_demand_m_s * factor
            number(submitted, nonnegative=True)
            self._update_calls += 1
            result = self._physical.update(self._claimant, frame.identity, submitted, frame.liquid_mm_h)
            after = self._physical.snapshot()
            require(isinstance(result, UpdateResult) and result.status == "SUCCESS" and result.identity == next_id and
                    after["identity"] == next_id, "physical update status/clock/owner differs")
            soil = number(result.soil_m, nonnegative=True)
            require(number(after["soil_m"], nonnegative=True) == soil, "physical update state/readback differs")
            self._identity, self._sentinel = next_id, decision.next_sentinel
            self._inputs = {}
            self._phase = "READY"
            return result
        except BaseException as error:
            # A physical update may already have changed its state. Keep it;
            # poison this trajectory instead of asserting rollback or retry.
            self._poison(error)
            raise

    def fork(self, trajectory_id, *, model=None, scaler_id=None):
        self._live()
        require(not self._inputs, "fork requires clean boundary")
        name(trajectory_id)
        require(trajectory_id != self._identity.trajectory_id and trajectory_id not in self._fork_ids, "duplicate branch trajectory")
        require(model is None or model == self._model, "changed model requires origin replay")
        require(scaler_id is None or scaler_id == self._scaler_id, "changed scaler requires origin replay")
        require(self._physical.snapshot()["identity"] == self._identity and self._sentinel.identity == self._identity,
                "unclean physical/neural clock boundary")
        new_id = replace(self._identity, trajectory_id=trajectory_id)
        child = object.__new__(Controller)
        child._identity, child._model, child._scaler_id = new_id, self._model, self._scaler_id
        child._policy, child._claimant = self._policy, object()
        child._physical = self._physical._fork(new_id, child._claimant)
        require(child._physical is not self._physical and child._physical.snapshot()["identity"] == new_id, "independent physical fork required")
        child._sentinel = Sentinel(new_id, tuple(list(self._sentinel.values)))
        child._inputs, child._fork_ids = {}, set()
        child._phase, child._failure = "READY", None
        child._policy_calls = child._update_calls = 0
        self._fork_ids.add(trajectory_id)
        return child

    def audit(self):
        # Diagnostics remain available after failure; this is not a restart API.
        return {"phase": self._phase, "failure": self._failure, "identity": self._identity,
                "sentinel": self._sentinel, "physical": self._physical.snapshot(),
                "pending_inputs": dict(self._inputs), "policy_calls": self._policy_calls,
                "physical_update_calls": self._update_calls, "rollback_claim": False}


@dataclass(frozen=True)
class ToyObjectiveSpec:
    origin_epoch: int
    input_id: str
    support_id: str
    scaler_id: str
    observations_or_native: bool = False

    def __post_init__(self):
        require(type(self.origin_epoch) is int and self.origin_epoch >= 0 and self.origin_epoch % HOUR == 0, "toy origin required")
        for value in (self.input_id, self.support_id, self.scaler_id): name(value)
        require(self.observations_or_native is False, "toy arithmetic only")


class ToyEvaluator:
    """One-use analytic quadratic. No physical simulator or observed loss."""
    def __init__(self, spec, target):
        require(isinstance(spec, ToyObjectiveSpec), "explicit toy objective specification required")
        require(type(target) is tuple and target and len(target) <= 16, "small immutable analytic target required")
        self.spec, self.target = spec, tuple(number(v) for v in target)
        self.calls, self.epoch, self.used = 0, spec.origin_epoch, False
        self.state = {"history": []}

    def evaluate(self, theta):
        require(not self.used and self.calls == 0 and self.epoch == self.spec.origin_epoch and not self.state["history"],
                "objective requires fresh independent origin state")
        self.used, self.calls = True, 1
        require(type(theta) is tuple and len(theta) == len(self.target), "toy parameter dimension differs")
        values = tuple(number(v) for v in theta)
        loss = math.fsum((v-t)**2 for v,t in zip(values, self.target))
        number(loss, nonnegative=True)
        self.state["history"].append(values)
        self.epoch += HOUR
        return loss


def symmetric_pair(theta, perturbation, radius, spec, factory):
    """Exactly one two-call value difference, with no optimizer/update loop."""
    require(type(theta) is tuple and 1 <= len(theta) <= 16, "small immutable theta required")
    theta = tuple(number(v) for v in theta)
    require(type(perturbation) is tuple and len(perturbation) == len(theta) and
            all(type(v) is int and v in (-1,1) for v in perturbation), "fixed plus/minus1 perturbation required")
    radius = number(radius)
    require(radius > 0 and isinstance(spec, ToyObjectiveSpec) and callable(factory), "positive radius and explicit toy contract required")
    plus = tuple(number(v + radius*d) for v,d in zip(theta,perturbation))
    minus = tuple(number(v - radius*d) for v,d in zip(theta,perturbation))
    evaluators = (factory(), factory())
    require(all(type(e) is ToyEvaluator for e in evaluators), "analytic toy evaluators only")
    a,b = evaluators
    require(a is not b and a.state is not b.state and a.state["history"] is not b.state["history"], "independent evaluator states required")
    require(a.target == b.target and all(e.spec == spec and not e.used and e.calls == 0 and
            e.epoch == spec.origin_epoch and not e.state["history"] for e in evaluators), "fresh identical objective input/support/origin required")
    j_plus = a.evaluate(plus)
    j_minus = b.evaluate(minus)
    require(all(e.used and e.calls == 1 and e.epoch == spec.origin_epoch+HOUR and e.spec == spec for e in evaluators),
            "one complete evaluation per independent origin required")
    estimate = tuple(number((j_plus-j_minus)/(2.0*radius*d)) for d in perturbation)
    return {"plus": plus, "minus": minus, "J_plus": j_plus, "J_minus": j_minus,
            "symmetric_estimate": estimate, "objective_calls": 2, "spec": spec,
            "optimizer_steps": 0, "analytic_toy_only": True, "native_gradient_claim": False}

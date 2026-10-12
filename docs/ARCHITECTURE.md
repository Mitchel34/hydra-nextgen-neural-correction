# Two correction boundaries

## Outlet residual model

The retained `gru_transformer` implementation consumes normalized history `[N,168,10]`, physical discharge forecasts `[N,18]`, error features `[N,18,7]` and site indices `[N]`. It returns normalized residuals `[N,18]`. Restoring discharge units, adding residuals to physical forecasts, checking physical output validity and verifying feature availability are caller responsibilities.

The history encoder projects to width 128, applies a residual GRU stage and one four-head pre-normalized full-attention Transformer block, then takes the final historical context. The prediction head combines that context with site/lead embeddings and forecast/error projections. V3 sees all 18 issued forecast leads and the declared error features. V0–V4 preserve controlled information variants; V4 adds identity-initialized FiLM. History attention is full attention over history already available at issue time; it does not authorize future observations. Recurrent state is local to each call.

Source shape checks are numerical guards. The bare model does not prove that masks, sources, scaling, site indices or issue-time vintages are scientifically valid. The original data loaders, training orchestration and fitted artifacts are intentionally outside this public boundary. The optional check verifies only two deterministic synthetic forward calls and dtype rejection on the GRU–Transformer V3 path.

## Internal process model

The internal kernels map three raw nonnegative features—potential evaporation demand in m/s, liquid input in mm/h and pre-update soil storage in m—through an explicit immutable scaler. The MLP has 16 hidden units and 81 parameters. The GRU has 32 hidden units and 3,489 parameters.

The reset-after GRU uses gate order R, Z, N:

```text
r = sigmoid(Wr x + Ur h + br)
z = sigmoid(Wz x + Uz h + bz)
n = tanh(Wn x + bn + r * (Un h))
candidate_h = (1-z) * n + z * h
head = head_w · candidate_h + head_b
factor = exp(log(2) * tanh(head))
```

The candidate recurrent bias is fixed at zero. This is not a stock MLX checkpoint format. A zero scalar head returns exactly 1; the engineering factor range is `[0.5,2]`. Negative normalized inputs and hidden values are valid even though raw features must be nonnegative.

C output is `[head, factor, hidden...]`; every error leaves output unchanged. Output/input overlap is rejected. Dots accumulate from zero in column order before adding bias; all used inputs and intermediates must remain finite. Compile without fast-math or floating-point contraction. Python serializes exact canonical JSON with feature, layout, scaler and model content identities; pickle is never used.

Returned GRU state is only a candidate. The [original C/BMI adapter source](../native/cfe_adapter/README.md) stages raw demand and liquid input, reads pre-update soil storage, and commits the candidate frame/hidden state only after the original physical update succeeds with the expected clock and finite storage. Failed or partial updates poison the wrapper; it does not pretend to roll back opaque physical state. The neutral head avoids a demand override.

The C artifact loader validates structure and fixed feature/layout IDs; it records model/scaler IDs and file-hash tags without cryptographically recomputing them. Native callers must verify canonical model contents and actual model/config hashes before initialization, around operations and after cleanup, and permanently reject a trajectory after drift. No enforcing native caller is supplied; see [required caller verification](BRIDGE.md#required-caller-verification).

This is source preservation, not qualified native execution. The external component must satisfy the expected variable, unit, scalar, clock and lifecycle contracts; its current deployed identity, state fidelity and conservation remain unverified. No CFE implementation, physical data, optimizer or trained internal corrector is supplied. The Python [process-boundary fixture](BRIDGE.md) models ownership and transaction order with an invented bucket and opaque sentinel, not CFE or the GRU.

## Evaluation

The dependency-free guards distinguish issue time, valid time and declared observation availability. A minimum observation delay is enforced. Scaling is limited to unique-hour fitting support. Stacked correction metadata must use disjoint chronological supports and a frozen upstream identity. Scoring requires the exact registered key set, rejects nonfinite candidate values and averages normalized RMSE equally across leads within gauge, then gauges. Missing observations remain masked. PBIAS is a paired sum-of-samples diagnostic, not an event-volume integral over overlapping issue trajectories.

These checks validate supplied metadata; they cannot establish historical publication vintages or authenticate a physical replay.

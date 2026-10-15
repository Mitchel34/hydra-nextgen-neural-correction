# Qualification and cost estimation

**Methods proposal — 15 October 2026.** This guide describes how to qualify and estimate a future neural-correction study. It records no new execution, fitted model, measured speedup or learned forecast benefit. The [current evidence boundaries](STATUS.md) continue to apply.

## Freeze the experiment before measuring it

Create a versioned recipe containing the issue windows, sites, retained leads, observation availability, feature definitions, units, missing-target rules and chronological fitting, selection and assessment supports. Bind input and executable identities by checksum. Record initialization, normalization fitted on permitted support, optimizer, search limits, seeds, stopping criteria and the rule for selecting a candidate.

Freeze an upstream corrector before generating data for a stacked residual model. Its fitting support must be disjoint from the downstream fitting support; selection and assessment data must retain their declared roles. Previously inspected research data remain exposed data. Renaming a window does not create an independent holdout.

## Proposed six-role continuation

The following seed-17 design is a proposal, not a report of six completed fits. Register exact chronological boundaries and dataset identities for A, B, C, D and E0 before use. Here `P` denotes the fixed physical reference; A fits internal correctors, B selects their candidates, C fits matched residual models, D selects residual checkpoints, and E0 is reserved for assessment subject to its actual exposure history. SPSA means simultaneous perturbation stochastic approximation.

| Role | Proposed fit and selection | Upstream forecast identity |
| --- | --- | --- |
| IF17 | MLP with hidden width 16; two SPSA updates on A; B selects center 0 or 2 | Physical trajectories with that candidate internal corrector |
| IG17 | GRU with hidden width 32; two SPSA updates on A; B selects center 0 or 2 | Physical trajectories with that candidate internal corrector |
| O-matched17 | Outlet residual fit on C; complete-epoch checkpoint selection on D | P |
| O-full17 | Outlet residual fit on the registered A+B+C union; complete-epoch selection on D | P |
| HF17 | Outlet residual fit on C; complete-epoch selection on D | Selected IF17's own forecasts |
| HG17 | Outlet residual fit on C; complete-epoch selection on D | Selected IG17's own forecasts |

Center 0 is the initial candidate and center 2 is the candidate after two updates. The proposed internal search budgets eight A objective evaluations across both roles (two perturbations per update) and four B selection evaluations (two centers per role). Each evaluation covers its declared support and can require many native issue trajectories; these counts are neither twelve trajectories nor twelve additional fits. Register trajectories per evaluation, the objective, perturbation and step-size schedules, normalization and failure rules before execution; symmetric perturbation arithmetic alone does not qualify this native optimization path.

All four residual roles use the proposed V3 GRU–Transformer recipe: 219,049 parameters, 168 × 10 history, 18 forecast leads and 18 × 7 mature error features; AdamW learning rate 0.001 and weight decay 0.0001; at most 20 epochs or 1,140 optimizer updates, whichever comes first; batch size 16, microbatch size 1 and patience of five complete epochs. Only complete epochs are eligible for D checkpoint selection. Feature availability and training transformations require independent qualification.

Freeze the selected IF17 and IG17 identities before producing each corrector's own C forecasts. Their D and E0 state/history and forecast-generation jobs remain separate work items. HF17 and HG17 must not substitute P's forecasts for their corrected upstream forecasts. O-full17 deliberately uses broader fitting support than O-matched17; report that difference rather than interpreting every contrast as an equal-data comparison.

Keep E0 labels unavailable to fitting and selection; release them for assessment only after model identities and E0 prediction checksums are fixed. A predeclared causal observation policy must separately govern any feature or state-history observations without relaxing that label seal; assessment targets cannot become adaptive inputs. Any prior E0 exposure must remain disclosed.

Count fitting, inference, forecast generation and scoring separately. A kernel inference call, physical time step, issue trajectory, whole-support objective evaluation and completed fit are distinct units. Internal fitting evaluates candidate correctors through the required native trajectories. A residual fit learns from already bound forecast/target pairs. Parameter count alone cannot predict either workload: physical replay, feature assembly and data movement may dominate.

## Reuse only the work whose identity is unchanged

| Artifact | Conditions for reuse | Reason to regenerate |
| --- | --- | --- |
| Fixed physical forecasts | Same inputs, issue schedule, model/configuration, initialization and execution semantics | Changed physical state, forcing, parameters or time convention |
| Internally corrected forecasts | Same frozen corrector, features, upstream state and native runtime in addition to the physical bindings | New corrector weights, feature normalization or state behavior |
| Residual features and targets | Same forecast identity, issue-available features, observation version, masks and units | Changed support, latency rule, transformation or observations |
| Scores | Same registered forecast keys, target masks, scales and aggregation | Changed support or comparison definition |

A cache hit must match the complete declared dependency identity, not just a filename. A new outlet residual model can reuse an unchanged upstream forecast cache. A new internal corrector generally changes the physical trajectory and needs corresponding forecast regeneration. Cache reuse never authorizes fitting on assessment data.

## Qualify in stages

1. **Generated fixtures:** check shapes, finite values, units at interfaces, serialization, neutral behavior and independent candidate state. Retain invalid-input and failure-path cases. These checks establish software behavior on invented examples.
2. **Native integration:** bind the actual executable and component versions; verify lifecycle, reset, branch ownership, commit-after-success behavior and failure recovery against a reference. Record unresolved outcomes rather than silently treating them as passes.
3. **Scientific behavior:** verify issue-time availability, normalization support, physical units, water/energy accounting where relevant, hidden-state consistency and restart/replay fidelity. A bounded correction factor does not establish conservation.
4. **Representative profiling:** measure a small, fully identified workload only after its required correctness gates pass. Include startup, cold-cache and warm-cache behavior, output handling and one bounded recovery exercise.
5. **Paired assessment:** freeze the final recipe before assessment and compare complete forecasts on shared keys, masks and scales using the [paired-comparison contract](PAIRED_COMPARISON.md). Keep fixed physical, outlet-only, internal-only and stacked comparisons interpretable.

Large-object handling needs its own qualification: declared size and hash, bounded memory, interrupted transfer/restart, duplicate prevention and recovery after incomplete writes. Success on a tiny artifact does not establish safe handling of a full forecast archive.

## Bind every measurement to its workload

Record the source revision, environment and machine shape, input hashes, cold/warm cache state, role, number of issues/sites/leads, native queries, optimizer updates, precision and parallelism. Separate successful units, rejected units, retries and incomplete units. State whether setup and teardown are included in wall time.

For each measured stage retain wall time, process CPU time, peak memory, bytes read/written, retained output size and verification outcome. Distinguish process memory from whole-machine use. Measure full issue trajectories when state or initialization makes isolated model calls unrepresentative.

Report a typical and a conservative duration from declared repetitions; keep startup and recovery overhead separate. A timing result without matching correctness and workload identities is insufficient for campaign sizing. Do not extrapolate a synthetic fixture into a real-data runtime claim.

## Estimate duration, cost and machine size

For stage `s` on machine option `j`, let `N_s` be remaining workload units and `t_s,j` the measured wall seconds per unit at the tested parallelism. Express setup and recovery in seconds, and `rate_j` in the selected currency per VM-hour:

```text
T_s,j = setup_s,j + N_s * t_s,j + recovery_s,j
H_j = sum of planned instance lifetimes on option j, in hours
estimated_cost = sum_j(rate_j * H_j)
                 + disk + retained_storage + requests + transfer + other_services
```

Use the same unit for each stage's `N` and measured `t`: whole-support objective evaluations or an explicit expansion into native issue trajectories for internal search, optimizer updates for residual fitting, and trajectories for forecast generation. Include generation for every changed upstream identity. State assumptions and compute a conservative scenario; never silently assume free retries or cache hits.

Choose the lowest estimated-cost option among those that meet the memory margin, compatibility checks, required completion time and correctness gates. Compare measured throughput at the actual parallelism. Doubling vCPUs does not establish a twofold speedup; a faster VM is cheaper only when its reduction in charged time outweighs its higher rate and ancillary costs.

An **illustrative profiling envelope** is at most **5 aggregate VM-hours and USD 8 total estimated cost**, stopping at whichever limit is reached first. Two VMs running for one hour consume two VM-hours. These figures are a proposed example, not a campaign estimate, account quote, spending authorization or record of a completed run. Reserve part of the envelope for setup, verification and controlled recovery before selecting the measurement size.

For a real estimate, record currency, region, machine type, purchase model, price retrieval date and all ancillary rates. Use the published [Compute Engine pricing](https://cloud.google.com/products/compute/pricing), [disk and image pricing](https://cloud.google.com/compute/disks-image-pricing) and [network pricing](https://cloud.google.com/vpc/network-pricing). The VM overview excludes several ancillary charges; this guide fixes no current hourly rate. Public list prices are planning inputs, not account-specific quotes. Sources checked 15 October 2026.

## Report the decision without promoting the evidence

Publish a compact methods/result summary with the frozen recipe identity, qualification outcomes, measured workload and resource totals, estimation assumptions and remaining gates. Retain failed and interrupted measurements in the evidence record. Separate measured quantities from extrapolations and label any continuation as proposed until it executes.

The [publication contract](PUBLICATION_CONTRACT.md) governs what enters this repository. Small sanitized methods, metrics and checksums can make the decision reviewable; operational records, bulk inputs and fitted weights remain outside this guide. A profiling result can support a resource decision. Learned hydrologic benefit requires the completed, qualified paired scientific assessment.

# Portable bridge code and its limits

This addition preserves original HYDRA research source rather than replacing it with campaign software. It has three distinct parts.

| Part | Preserved behavior | Check scope |
| --- | --- | --- |
| `bridge_artifacts.py` | Exact canonical JSON ↔ little-endian binary encoding, model/scaler/feature/layout content identities, fresh-file export and caller-bound verification | Generated MLP/GRU artifacts, malformed headers, drift, nonfinite payloads, symlinks and no-overwrite behavior |
| `process_boundary.py` | Input staging, pre-update features, immutable ownership, candidate-state commit after success, failure poisoning and independent clean branches | Invented `MockPhysical` bucket and `Sentinel`; no neural network or native hydrologic model |
| `native/cfe_adapter/neural_bridge.c/.h` | Original C/BMI demand wrapper with strict artifact loading and transaction guards | Source syntax against an exact external BMI header and rejection of unsafe-math compilation; no linking/loading/physical calls |

## Artifact format

The codec uses magic `HYDRANN1`, version 1, and a 300-byte header encoded as `<8s9I64s64s64s64s`. The header fixes architecture, three inputs, hidden width, parameter count, float64 dtype, GRU gate convention and payload count. Four lowercase SHA-256 identities bind feature order, parameter layout, model and scaler. The payload contains three means, three strictly positive scales and the flat original parameters as little-endian binary64.

| Architecture | Hidden width | Parameters | Total binary bytes |
| --- | --- | --- | --- |
| MLP | 16 | 81 | 996 |
| Reset-after GRU | 32 | 3,489 | 28,260 |

Exact EOF and a canonical round trip are required. The package pins the unchanged Python inference reference by content hash. `export_files` accepts two distinct fresh absolute paths supplied by the caller and uses exclusive creation; an interrupted export may leave the first newly created file and does not claim an atomic two-file transaction. `verify_files` rejects symlinked artifacts and verifies caller-supplied identities and sizes. No trained artifact is checked in.

## Synthetic process/state boundary

The fixture stages raw demand in m/s and liquid input in mm/h, captures pre-update storage in m, and permits one decision per successful hourly physical step. Only a successful update with the expected clock and finite readback commits its opaque sentinel. A failed or partially advanced update poisons the controller; the fixture does not invent a rollback. Model/scaler changes require a fresh origin, and clean branches retain independent state.

The sentinel is an ownership token with 32 coordinates, not the GRU equations. `MockPhysical` is an invented algebraic bucket, not CFE. `ToyEvaluator` and symmetric perturbations test toy objective/support ownership and algebra, not a trained optimizer or a scientifically qualified gradient.

## Native adapter source

The adapter source is copied byte for byte from the original HYDRA prototype. It calls an external BMI component rather than embedding CFE algorithms. Its source includes configuration parsing and dynamic-library loading for future caller-controlled integration. Those functions are not called by the public checks, and no runtime configuration, model weights, physical inputs or private campaign route is provided.

The [adapter guide](../native/cfe_adapter/README.md) and [external dependency manifest](../native/cfe_adapter/DEPENDENCIES.json) record the pinned interface and upstream notices. Syntax success cannot establish component authenticity, physical closure, native branch identity, hidden-state equivalence or portable restart. The C baseline results remain physical-reference evidence only; these additions introduce no paired learned-correction claim.

Exact source identities and the two packaging changes are recorded in [BRIDGE_SOURCE_PROVENANCE.json](BRIDGE_SOURCE_PROVENANCE.json). The original inference/evaluation/outlet mathematics and retained third-party license texts remain unchanged.

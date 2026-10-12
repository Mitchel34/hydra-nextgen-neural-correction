# Portable bridge code and its limits

This addition preserves original HYDRA research source rather than replacing it with campaign software. It has three distinct parts.

| Part | Preserved behavior | Check scope |
| --- | --- | --- |
| `bridge_artifacts.py` | Exact canonical JSON ↔ little-endian binary encoding, model/scaler/feature/layout content identities, fresh-file export and caller-bound verification | Generated MLP/GRU artifacts, malformed headers, drift, nonfinite payloads, symlinks and no-overwrite behavior |
| `process_boundary.py` | Input staging, pre-update features, immutable ownership, candidate-state commit after success, failure poisoning and independent clean branches | Invented `MockPhysical` bucket and `Sentinel`; no neural network or native hydrologic model |
| `native/cfe_adapter/neural_bridge.c/.h` | Original C/BMI demand wrapper with structural artifact checks and transaction guards; caller must verify content/hashes | Source syntax against an exact external BMI header and rejection of unsafe-math compilation; no linking/loading/physical calls |

## Artifact format

The codec uses magic `HYDRANN1`, version 1, and a 300-byte header encoded as `<8s9I64s64s64s64s`. The header fixes architecture, three inputs, hidden width, parameter count, float64 dtype, GRU gate convention and payload count. Four lowercase hexadecimal fields carry SHA-256 identities for feature order, parameter layout, model and scaler. The Python codec recomputes and checks these content identities; the C loader independently checks only the fixed feature/layout IDs and records the model/scaler IDs as declared tags. The payload contains three means, three strictly positive scales and the flat original parameters as little-endian binary64.

| Architecture | Hidden width | Parameters | Total binary bytes |
| --- | --- | --- | --- |
| MLP | 16 | 81 | 996 |
| Reset-after GRU | 32 | 3,489 | 28,260 |

The Python codec requires exact EOF and a canonical JSON/binary round trip. The package pins the unchanged Python inference reference by content hash. `export_files` accepts two distinct fresh absolute paths supplied by the caller and uses exclusive creation; an interrupted export may leave the first newly created file and does not claim an atomic two-file transaction. `verify_files` rejects symlinked artifacts, recomputes the actual JSON/binary SHA-256 values against the caller binding, and checks canonical content identities, exact JSON/binary agreement and sizes. It is a stateless model-file check; it does not verify CFE configuration files or enforce a native trajectory lifecycle. No trained artifact is checked in.

## Synthetic process/state boundary

The fixture stages raw demand in m/s and liquid input in mm/h, captures pre-update storage in m, and permits one decision per successful hourly physical step. Only a successful update with the expected clock and finite readback commits its opaque sentinel. A failed or partially advanced update poisons the controller; the fixture does not invent a rollback. Model/scaler changes require a fresh origin, and clean branches retain independent state.

The sentinel is an ownership token with 32 coordinates, not the GRU equations. `MockPhysical` is an invented algebraic bucket, not CFE. `ToyEvaluator` and symmetric perturbations test toy objective/support ownership and algebra, not a trained optimizer or a scientifically qualified gradient.

## Native adapter source

The adapter source is copied byte for byte from the original HYDRA prototype. It calls an external BMI component rather than embedding CFE algorithms. Its source includes configuration parsing and dynamic-library loading for future caller-controlled integration. Those functions are not called by the public checks, and no runtime configuration, model weights, physical inputs or private campaign route is provided.

The C loader checks bounded regular-file reads, exact structural format and dimensions, finite values, positive scales, and the fixed feature/layout IDs. It does not recompute model/scaler content IDs or actual model/config SHA-256 values; those fields are recorded as declared provenance tags. A finite parameter change with unchanged declared tags can therefore pass the C loader. Structural success is not content authentication.

### Required caller verification

The original integration contract requires a caller to verify canonical model JSON against binary contents and actual model/config byte hashes against frozen expected bindings before initialization, before and after operations, and after cleanup. This covers model JSON, model binary, the physical CFE configuration and the adapter configuration. The caller must reject/poison the trajectory permanently after any drift; restoring a changed file does not permit reuse or hot reload. A model, scaler or configuration change requires a fresh physical/neural origin and context.

**No enforcing native caller is supplied in this public package.** The Python codec provides stateless model-content checks, and the synthetic boundary tests exercise separate toy ownership behavior. Neither connects those checks to the C/BMI lifecycle, verifies its configuration hashes around operations, or enforces native drift rejection. Anyone implementing a native caller must supply that verification and qualify the full integration separately.

The [adapter guide](../native/cfe_adapter/README.md) and [external dependency manifest](../native/cfe_adapter/DEPENDENCIES.json) record the pinned interface and upstream notices. Syntax success cannot establish component authenticity, physical closure, native branch identity, hidden-state equivalence or portable restart. The C baseline results remain physical-reference evidence only; these additions introduce no paired learned-correction claim.

Exact source identities and the two packaging changes are recorded in [BRIDGE_SOURCE_PROVENANCE.json](BRIDGE_SOURCE_PROVENANCE.json). The original inference/evaluation/outlet mathematics and retained third-party license texts remain unchanged.

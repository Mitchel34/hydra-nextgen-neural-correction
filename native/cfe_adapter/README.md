# Original HYDRA C/BMI adapter source

`neural_bridge.c` and `neural_bridge.h` are original HYDRA source, copied byte for byte. They depend on the existing HYDRA `native/kernels.c/.h` and an external `bmi.h`; no upstream CFE code or BMI header is vendored. Source rights and changes are recorded in [bridge provenance](../../docs/BRIDGE_SOURCE_PROVENANCE.json). The project has no newly assigned license.

The adapter stages demand/liquid features, records pre-update soil storage, applies a bounded immutable-model demand factor, and forwards one update to the original BMI component. Candidate recurrent state and the frame commit only after successful physical update, expected clock and finite storage. The neutral head avoids overriding the original demand. Failed or partial physical updates poison the wrapper; the adapter cannot roll back opaque CFE state.

The retained source requires POSIX dynamic-loading/process interfaces and ordinary binary64 math. Its original configuration/artifact parsing and lifecycle assumptions are research integration code. No build/runtime configuration or native campaign launcher is supplied, and no end-to-end CFE qualification is claimed.

## Structural loading and caller verification

The C loader checks format, dimensions, finite values, positive scales and fixed feature/layout IDs. It records model/scaler IDs and `model_file_sha256`/`config_sha256` as declared tags without recomputing their content or file hashes. A finite parameter change retaining those tags can pass its structural checks.

The required caller contract is to verify canonical JSON↔binary model contents and actual model/config SHA-256 values against frozen bindings before initialization, before and after operations, and after cleanup. Drift must permanently reject/poison that caller trajectory; restoring files cannot authorize reuse or hot reload. A changed model, scaler or configuration requires a fresh physical/neural origin. See [required caller verification](../../docs/BRIDGE.md#required-caller-verification).

No enforcing native caller is included. The Python model codec is stateless, and the public compiler check neither verifies model/config files around a native operation nor exercises drift rejection.

## Bounded source check

Supply the exact external header identified in [DEPENDENCIES.json](DEPENDENCIES.json):

```sh
python scripts/check_adapter_source.py --bmi-include /path/to/external/include
```

Run from the repository root with a local C11 compiler. The checker verifies the header hash, invokes `-fsyntax-only`, and confirms that `-ffast-math` triggers the original finite-math guard. It produces no binary, links or loads no CFE library, and executes no hydrologic model. A changed upstream header is rejected rather than silently treated as equivalent.

See [the bridge guide](../../docs/BRIDGE.md) and [recorded validation](../../results/bridge_checks_2026_10_12.json). Native lifecycle, conservation, state/replay fidelity and learned hydrologic skill remain separate research requirements.

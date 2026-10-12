"""Recompute the published baseline report without private data or a model run."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from hydra_correctors.baseline_report import validate_bundle


if __name__ == "__main__":
    print(json.dumps(validate_bundle(root / "results/c_physical_baseline_2019_10"), indent=2, sort_keys=True))

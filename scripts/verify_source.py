"""Verify the public bytes of the extracted and newly authored source indices."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifests = ["docs/SOURCE_PROVENANCE.json", "docs/CONSOLIDATION_PROVENANCE.json",
             "docs/BRIDGE_SOURCE_PROVENANCE.json"]
entries = []
for name in manifests:
    entries.extend(json.loads((root / name).read_text())["files"])
for entry in entries:
    path = root / entry["public_path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit("Invalid provenance path")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["published_sha256"]:
        raise SystemExit("Published source differs: " + entry["public_path"])
print(f"Verified {len(entries)} provenance-bound files.")

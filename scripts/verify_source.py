"""Verify the public bytes of every explicitly extracted source file."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/SOURCE_PROVENANCE.json").read_text())
for entry in manifest["files"]:
    path = root / entry["public_path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit("Invalid provenance path")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["published_sha256"]:
        raise SystemExit("Published source differs: " + entry["public_path"])
print(f"Verified {len(manifest['files'])} extracted files.")

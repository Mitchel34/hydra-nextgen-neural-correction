"""Generated immutable-model artifacts only; no fitted weights or native calls."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from hydra_correctors import kernels as ref
from hydra_correctors import bridge_artifacts as bridge


def model(architecture):
    count = ref.layout(architecture)["parameter_count"]
    parameters = tuple((index % 7 - 3) * 0.001 for index in range(count))
    return ref.Model(architecture, parameters, ref.Scaler((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))


class BridgeArtifactTests(unittest.TestCase):
    def test_generated_mlp_and_gru_round_trip_and_independent_wire_header(self):
        for architecture, arch, count, size in [("mlp16", 1, 81, 996), ("gru32", 2, 3489, 28260)]:
            with self.subTest(architecture=architecture):
                current = model(architecture)
                raw = bridge.export_bytes(ref.dumps(current))
                self.assertEqual(len(raw), size)
                fields = struct.unpack("<8s9I64s64s64s64s", raw[:300])
                self.assertEqual(fields[:4], (b"HYDRANN1", 1, 300, arch))
                self.assertEqual(fields[4:9], (3, 16 if arch == 1 else 32, count, 1, 0 if arch == 1 else 1))
                self.assertEqual(fields[9], count + 6)
                self.assertEqual(struct.unpack("<6d", raw[300:348]), (0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
                self.assertEqual(bridge.decode_bytes(raw), current)
                self.assertEqual(bridge.export_bytes(ref.dumps(bridge.decode_bytes(raw))), raw)

    def test_truncation_extra_bytes_and_header_convention_drift_fail(self):
        raw = bridge.export_bytes(ref.dumps(model("gru32")))
        for mutated in [raw[:299], raw[:-1], raw + b"x", b"BADMAGIC" + raw[8:]]:
            with self.subTest(size=len(mutated)), self.assertRaises(ValueError):
                bridge.decode_bytes(mutated)
        fields = list(bridge.HEADER.unpack(raw[:300]))
        for index, value in [(1, 2), (2, 299), (3, 3), (4, 4), (5, 31), (6, 3488), (7, 2), (8, 0), (9, 3494)]:
            edited = fields.copy()
            edited[index] = value
            with self.subTest(field=index), self.assertRaises(ValueError):
                bridge.decode_bytes(bridge.HEADER.pack(*edited) + raw[300:])

    def test_changed_payload_or_identity_cannot_be_relabelled(self):
        raw = bridge.export_bytes(ref.dumps(model("mlp16")))
        for offset in [44, 108, 172, 236, 348]:
            edited = bytearray(raw)
            edited[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                bridge.decode_bytes(bytes(edited))

    def test_nonfinite_wire_data_and_noncanonical_json_are_rejected(self):
        raw = bytearray(bridge.export_bytes(ref.dumps(model("mlp16"))))
        raw[300:308] = struct.pack("<d", float("nan"))
        with self.assertRaises(ValueError):
            bridge.decode_bytes(bytes(raw))
        original = ref.dumps(model("mlp16"))
        with self.assertRaises(ValueError):
            bridge.export_bytes(original + b" ")

    def test_fresh_files_verify_and_existing_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path, binary_path = Path(folder) / "model.json", Path(folder) / "model.bin"
            current = model("gru32")
            binding = bridge.export_files(ref.dumps(current), json_path, binary_path)
            self.assertEqual(bridge.verify_files(binding), current)
            original = (json_path.read_bytes(), binary_path.read_bytes())
            with self.assertRaises(ValueError):
                bridge.export_files(ref.dumps(current), json_path, binary_path)
            self.assertEqual(original, (json_path.read_bytes(), binary_path.read_bytes()))

    def test_stale_hash_false_binding_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path, binary_path = Path(folder) / "model.json", Path(folder) / "model.bin"
            binding = bridge.export_files(ref.dumps(model("mlp16")), json_path, binary_path)
            for key, value in [("wire_version", True), ("binary_bytes", 997), ("architecture", "gru32")]:
                changed = deepcopy(binding)
                changed[key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    bridge.verify_files(changed)
            link = Path(folder) / "linked.bin"
            link.symlink_to(binary_path)
            changed = deepcopy(binding)
            changed["binary"]["path"] = str(link)
            with self.assertRaises(ValueError):
                bridge.verify_files(changed)
            binary_path.write_bytes(binary_path.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                bridge.verify_files(binding)


if __name__ == "__main__":
    unittest.main()

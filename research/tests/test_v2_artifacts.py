from pathlib import Path

from brazil_rv.v2.artifacts import inventory, sha256_file, verify_inventory, write_json_atomic


def test_deterministic_json_and_inventory(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    digest_a = write_json_atomic(first, {"z": 1, "a": [2, 3]}, write_sha256=False)
    digest_b = write_json_atomic(second, {"a": [2, 3], "z": 1}, write_sha256=False)
    assert first.read_bytes() == second.read_bytes()
    assert digest_a == digest_b == sha256_file(first)

    rows = inventory(tmp_path)
    verify_inventory(tmp_path, rows)

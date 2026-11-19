"""The frozen v2 splits never change. FREEZE.json's own sha256 is pinned here; every manifest's sha256 must match
FREEZE.json. Changing a split, a manifest or the freeze record fails this test."""
import hashlib
import json

from ap_sim.generator.splits import DIR, SPLITS

# Re-pinned 2026-10-31 for a change to interfaces/tools.py only (the record's `amendments` entry); was
# aeb6329d220127a0ee3d6068512cb85d96c85f442317c1ed36493ef1ed574fb3. No split file changed.
FREEZE_SHA256 = "d17ac59ae138f9561d7c551275a90b5b2c12fde9c16b15676ef3fbd010b8e19c"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_freeze_record_is_the_committed_one():
    assert sha(DIR / "FREEZE.json") == FREEZE_SHA256


def test_every_frozen_file_matches_its_hash():
    freeze = json.loads((DIR / "FREEZE.json").read_text())
    assert set(freeze["files"]) == {f"{s}.json" for s in SPLITS} | {"build_summary.json"}
    for name, want in freeze["files"].items():
        assert sha(DIR / name) == want, name


def test_no_unfrozen_manifest_in_the_directory():
    assert {p.name for p in DIR.glob("*.json")} == set(json.loads((DIR / "FREEZE.json").read_text())["files"]) | {
        "FREEZE.json"}

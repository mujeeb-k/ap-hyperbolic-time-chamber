"""The v2 split loaders accept only their own split.

- Each loader accepts its own frozen manifest, in frozen order, and nothing else: another split's manifest, an entry
  moved in from another split (caught by its keys, whatever its label), a changed file.
- SFT, RL, dev, regression and test refuse any case with a reversal, including one whose label hides it, and
  any sandbox-v2 case. The V6 loader accepts only cases that are V6 by content.
- The test loader gives 200 cases, or 300 with the reserve appended in frozen order; the Claude row's 100 are the
  first 100 test cases.
- Every case regenerates identically from its seed and forced variables, and its stored labels are the grader's."""
import copy
import json

import pytest

from ap_sim.generator import causal, policy_shift
from ap_sim.generator.manifests import _tuples, params_sha
from ap_sim.generator.splits import (DIR, SPLITS, TRAINING, SplitRefused, first_test_cases, gap_at_start, is_v6,
                                     keys, load_split, load_test)
from ap_sim.grader.grader import grade
from ap_sim.interfaces.structured import StructuredSurface
from ap_sim.interfaces.tools import ToolLayer

SIZES = {"sft": 300, "rl": 480, "dev": 60, "regression": 40, "test": 200, "test_reserve": 100, "v6": 100,
         "policy_shift": 44}


def manifest(split):
    return json.loads((DIR / f"{split}.json").read_text())


def moved(entry, split, **changes):
    """An entry relabelled as `split`'s own (as a mislabelling or a copy would look)."""
    return {**copy.deepcopy(entry), "split": split, **changes}


def with_entry(split, entry):
    m = manifest(split)
    m["entries"] = m["entries"][:3] + [entry]
    return m


@pytest.mark.parametrize("split", SPLITS)
def test_each_loader_accepts_its_own_frozen_split_in_order(split):
    f = load_split(split)
    assert len(f) == SIZES[split] and list(f) == [e["case_id"] for e in manifest(split)["entries"]]


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("other", SPLITS)
def test_each_loader_refuses_every_other_split(split, other):
    if split == other:
        return
    with pytest.raises(SplitRefused):                                   # the other manifest itself
        load_split(split, manifest=manifest(other))
    entry = manifest(other)["entries"][0]
    with pytest.raises(SplitRefused):                                   # one of its cases, relabelled as ours
        load_split(split, manifest=with_entry(split, moved(entry, split)))


@pytest.mark.parametrize("split", TRAINING + ("test",))
def test_training_and_test_loaders_refuse_a_mislabelled_v6_case(split):
    e = manifest("v6")["entries"][0]
    label = {"variant": "V1_delivered_unrecorded", "group": "V1"}
    with pytest.raises(SplitRefused, match="reversal|v6/"):
        load_split(split, manifest=with_entry(split, moved(e, split, **label)))


@pytest.mark.parametrize("split", TRAINING + ("test", "test_reserve"))
def test_any_reversal_is_refused_even_a_covered_one_never_seen_before(split):
    c = causal.generate("X-covered-reversal", 12_345_678, {"gap": 0, "reversal": True, "stale_receipt": True,
                                                            "n_receipts": 2})
    assert c.params["reversals"] and gap_at_start(c.params) == 0 and not is_v6(c.params)   # labelled V5, not V6
    params = {k: (list(map(list, v)) if k == "reversals" else list(v) if isinstance(v, tuple) else v)
              for k, v in c.params.items()}
    e = {"case_id": c.case_id, "split": split, "seed": c.seed, "params": params,
         "params_sha256": params_sha(_tuples(params)), "variant": c.variant}
    with pytest.raises(SplitRefused, match="reversal"):
        load_split(split, manifest=with_entry(split, e))
    with pytest.raises(SplitRefused, match="not V6 by content"):     # nor does the V6 loader take it
        load_split("v6", manifest=with_entry("v6", {**e, "split": "v6"}))


@pytest.mark.parametrize("split", TRAINING + ("test",))
def test_sandbox_v2_cases_are_refused(split):
    e = next(x for x in manifest("policy_shift")["entries"] if x["policy"] == "sandbox-v2")
    with pytest.raises(SplitRefused, match="sandbox-v2|policy_shift/"):
        load_split(split, manifest=with_entry(split, moved(e, split)))


def test_a_changed_manifest_is_refused(tmp_path):
    for split in SPLITS:
        (tmp_path / f"{split}.json").write_text((DIR / f"{split}.json").read_text())
    with open(tmp_path / "test.json", "a") as fh:
        fh.write(" ")
    import ap_sim.generator.splits as sp
    original = sp._frozen_hashes
    sp._frozen_hashes = lambda: json.loads((DIR / "FREEZE.json").read_text())["files"]
    try:
        load_split("sft", directory=tmp_path)
        with pytest.raises(SplitRefused, match="not the frozen one"):
            load_split("test", directory=tmp_path)
    finally:
        sp._frozen_hashes = original


def test_test_loader_sizes_and_the_claude_rows_first_100():
    base, ext = load_test(), load_test(extended=True)
    assert len(base) == 200 and len(ext) == 300
    assert list(ext)[:200] == list(base)
    assert list(ext)[200:] == [e["case_id"] for e in manifest("test_reserve")["entries"]]
    assert first_test_cases(100) == list(base)[:100]


def test_no_key_is_shared_across_splits():
    seen = {}
    for split in SPLITS:
        for e in manifest(split)["entries"]:
            k = keys(e["seed"], _tuples(e["params"]))
            assert k == e["keys"]
            for kind in ("exact", "near"):
                assert seen.setdefault(k[kind], split) == split
            assert seen.setdefault("g:" + k["group"], split) == split


def test_v6_split_is_v6_by_content_and_no_other_split_has_a_reversal():
    assert all(is_v6(e["params"]) for e in manifest("v6")["entries"])
    for split in SPLITS:
        if split != "v6":
            assert not any(e["params"]["reversals"] for e in manifest(split)["entries"])


@pytest.mark.parametrize("split", [s for s in SPLITS if s != "policy_shift"])
def test_every_case_regenerates_identically_from_its_seed_and_forcing(split):
    for e in manifest(split)["entries"]:
        c = causal.generate(e["case_id"], e["seed"], e["force"])
        assert params_sha(c.params) == e["params_sha256"] and c.variant == e["variant"]


def test_policy_shift_regenerates_from_its_seeds():
    m = manifest("policy_shift")
    want = {e["case_id"]: e["params_sha256"] for e in m["entries"]}
    got = {}
    s = m["seeds"]["pairs"]
    while len(got) < 40:
        p = policy_shift.pair(s)
        s += 1
        if p and policy_shift.expected(p[0]) == "keep_blocked" and policy_shift.expected(p[1]) == "release":
            got.update({c.case_id: params_sha(c.params) for c in p})
    for v1, v2, _ in policy_shift.boundary(m["seeds"]["boundary"]):
        got.update({c.case_id: params_sha(c.params) for c in (v1, v2)})
    assert got == want


@pytest.mark.parametrize("split", SPLITS)
def test_stored_labels_are_the_graders(split):
    for cid, f in load_split(split).items():
        e = next(x for x in manifest(split)["entries"] if x["case_id"] == cid)
        s = f()
        v = grade(s, ToolLayer(s, StructuredSurface(), "surfaces/4").episode_log())
        assert (v.expected, v.diagnostics["discrepancy_type"]["required"]) == (e["expected"], e["discrepancy_type"])

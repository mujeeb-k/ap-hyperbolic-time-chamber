"""The messiness stress set (manifests/stress/SPEC.md): builder, content predicates, design, manifest, loader."""
from __future__ import annotations

import collections
import itertools
import json

import pytest

from ap_sim.stress import messiness as st
from ap_sim.generator.fixtures import SCENARIOS, DEV_ONLY, build as frozen_build
from ap_sim.generator.manifests import _tuples
from ap_sim.generator.splits import DIR, keys

MANIFEST_PRESENT = st.MANIFEST.exists()
needs_manifest = pytest.mark.skipif(not MANIFEST_PRESENT, reason="manifests/stress/messiness.json not built")


def _timed(p: dict) -> dict:
    """A frozen-format parameter set (no reversals) in the stress format, with fixtures.build's timing."""
    rt = [-2400 + 400 * (i + 1) for i in range(len(p["receipts"]))]
    t, others = rt[-1], []
    for q, cancelled in [(q, True) for q in p.get("cancelled_invoices", ())] + [(q, False) for q in p["other_invoices"]]:
        t += 120
        others.append([q, t, p.get("cancelled_at") if cancelled else None])
    return dict(po_qty=p["po_qty"], price=p["price"], invoice_qty=p["invoice_qty"], blocked_at=p.get("blocked_at", -160),
                physical_total=p["physical_total"], delay=p["delay"],
                timed_receipts=[[q, x] for q, x in zip(p["receipts"], rt)], timed_other_invoices=others)


def test_frozen_format_is_fixtures_build_and_complications_read_its_timing():
    p = dict(po_qty=200, price="9.99", receipts=[40, 30], other_invoices=[25, 18], invoice_qty=25, physical_total=70,
             delay=240, cancelled_invoices=[18], blocked_at=-160, cancelled_at=-80, reversals=[])
    _same(st.build("x", 5, **p), frozen_build("x", 5, **_tuples(p)))
    assert st.complications(p) == ["M1", "M3", "M4"]
    assert st.timed_view(p)["other_invoices"] == [[18, -1480, -80], [25, -1360, None], [18, -1240, None]]


def _same(a, b) -> None:
    assert [r.model_dump() for r in a.records] == [r.model_dump() for r in b.records]
    assert a.world == b.world and a.current_invoice == b.current_invoice and a.policy == b.policy


def test_builder_reproduces_fixtures_build_on_frozen_timing():
    """POLICY D1/E1's V1 and the dev-only G7 (a cancellation after the block), as fixtures.SCENARIOS and DEV_ONLY
    build them."""
    v1 = dict(po_qty=100, price="20.00", receipts=[40, 30], other_invoices=[50], invoice_qty=30, physical_total=80,
              delay=480)
    g7 = dict(po_qty=90, price="18.40", receipts=[50, 25], other_invoices=[20], invoice_qty=50, physical_total=75,
              delay=480, cancelled_invoices=(40,), cancelled_at=-80)
    _same(st.build("V1", 101, **_timed(v1)), SCENARIOS["V1"]())
    _same(st.build("G7", 707, **_timed(g7)), DEV_ONLY["G7"]())
    _same(st.build("G7", 707, **_timed(g7)), frozen_build("G7", 707, **g7))


def test_builder_reproduces_the_test_split():
    """Every test-split case (frozen format, no reversal) built by the stress builder with the frozen timing is the
    same scenario, record for record."""
    for e in json.loads((DIR / "test.json").read_text())["entries"][:60]:
        p = _tuples(e["params"])
        assert not p.get("reversals")
        _same(st.build(e["case_id"], e["seed"], **_timed(p)), frozen_build(e["case_id"], e["seed"], **p))


BASE = dict(po_qty=300, price="9.99", invoice_qty=40, blocked_at=-160, physical_total=60, delay=240)


@pytest.mark.parametrize("receipts,others,want", [
    ([[60, -2000]], [], []),
    ([[30, -2000], [30, -1600]], [], ["M1"]),
    ([[60, -2000]], [[10, -1900, -1800], [12, -1850, -1700]], ["M2"]),
    ([[60, -2000]], [[10, -1900, -1800], [12, -1850, -1800]], []),          # same cancellation time: not M2
    ([[60, -2000]], [[10, -1900, -1800]], []),                               # one cancellation: not M2
    ([[60, -2000]], [[10, -1900, -1800], [10, -1700, None]], ["M3"]),       # re-issued after the cancellation
    ([[60, -2000]], [[10, -1900, -100], [10, -1700, None]], ["M3"]),        # re-issued before it
    ([[60, -2000]], [[10, -1700, None], [10, -1900, -1800]], ["M3"]),       # order of the list does not matter
    ([[60, -2000]], [[10, -1950, None], [10, -1900, -1800]], []),           # active posted before: not a re-issue
    ([[60, -2000]], [[40, -1900, None]], ["M4"]),
    ([[30, -2000], [40, -1600]], [[40, -1500, None], [9, -1400, -1300], [8, -1350, -1250]], ["M1", "M2", "M4"]),
])
def test_complications_by_content(receipts, others, want):
    assert st.complications({**BASE, "timed_receipts": receipts, "timed_other_invoices": others}) == want


def test_timing_rules():
    good = {**BASE, "timed_receipts": [[60, -2000]], "timed_other_invoices": [[10, -1900, -1800]]}
    assert st.timing_problems(good) == []
    for bad in ({**good, "timed_receipts": [[60, -1990]]}, {**good, "timed_other_invoices": [[10, -1900, -1950]]},
                {**good, "timed_other_invoices": [[10, -1900, -10]]},
                {**good, "timed_other_invoices": [[10, -2100, None]]},
                {**good, "timed_other_invoices": [[10, -2000, None]]},
                {**good, "timed_receipts": [[5, -2400 + 400 * i] for i in range(1, 7)]}):
        assert st.timing_problems(bad)


def test_timed_spacing_rule_is_enforced():
    """SPEC.md section 3 rule 5 (set on 2026-11-06): a timed case with two events less than 20 minutes apart,
    or an invoice posted after the block, is refused by the rules, so `generate` resamples it and `load` refuses it."""
    good = {**BASE, "timed_receipts": [[60, -2000], [30, -1600]], "timed_other_invoices": [[10, -1900, -1580]]}
    assert st.timing_problems(good) == []
    for bad in ({**good, "timed_other_invoices": [[10, -1900, -1595]]},     # 5 minutes after the receipt at -1600
                {**good, "timed_other_invoices": [[10, -1900, -1610]]},     # 10 minutes before it
                {**good, "timed_other_invoices": [[10, -1900, -170]]},      # 10 minutes before the block at -160
                {**good, "timed_other_invoices": [[10, -100, None]]}):      # posted after the block
        assert st.timing_problems(bad)
    # the two draws an independent check found breaking the rule (stress-V1-0, stress-V2-12): their seeds now resample to compliant cases
    for group, j, k in (("V1", 0, 0), ("V2", 12, 12)):
        gi = st.GROUPS.index(group)
        c = st.generate(f"stress-{group}-{k}", 11_800_000 + 10_000 * gi + k, group, j, st.design()[group][j])
        assert st.timing_problems(c.params) == [] and c.resamples >= 1


@needs_manifest
def test_frozen_spacing_rule_is_enforced(manifest, tmp_path):
    """The 20-minute spacing applies to frozen-format cases too (SPEC.md section 3 rule
    5). stress-V5-13 (frozen) with its cancellation 10 minutes after the block at -160 was accepted by the loader."""
    doc = json.loads(json.dumps(manifest))
    e = next(x for x in doc["entries"] if x["case_id"] == "stress-V5-13")
    assert not st.is_timed(e["params"]) and st.timing_problems(e["params"]) == []
    bad = {**e["params"], "cancelled_at": -150}
    assert any("minutes apart" in w for w in st.timing_problems(bad))
    # the generator's screen of a frozen-shaped M3 draw keeps the rules from before the 2026-11-06 changes
    assert not any("minutes apart" in w for w in st.timing_problems(bad, reissue_rule=False, frozen_spacing=False))
    e["params"], e["params_sha256"] = bad, st.params_sha(bad)
    doc["entries"] = [e]
    p = tmp_path / "messiness.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(st.StressRefused, match="minutes apart"):
        st.load(p)


def _spacing(p: dict) -> int:
    v = st.timed_view(p)
    ts = sorted([t for _, t in v["receipts"]] + [v["blocked_at"]]
                + [x for _, q, c in v["other_invoices"] for x in (q, c) if x is not None])
    return min(y - x for x, y in zip(ts, ts[1:]))


@needs_manifest
def test_every_manifest_case_meets_the_spacing_rule(manifest):
    """All 100 cases: every recorded event at least 20 minutes from every other (frozen-format cases too, though
    their timing is fixtures.build's), and no timed case posts an invoice after the block."""
    for e in manifest["entries"]:
        p = e["params"]
        assert _spacing(p) >= st.MIN_GAP_MINUTES, e["case_id"]
        if st.is_timed(p):
            assert all(q <= p["blocked_at"] for _, q, _ in p["timed_other_invoices"]), e["case_id"]


def test_design_swaps_keep_every_cell_total_and_avoid_infeasible_v5_cells():
    a, d = st.allocation(), st.design_allocation()
    for c in st.cells():
        assert sum(a[g][c] for g in st.GROUPS) == sum(d[g][c] for g in st.GROUPS)
    assert d["V5"][("M3",)] == 0 and d["V5"][("M3", "M4")] == 0     # no stale cause without M1 or M2
    moved = sum(abs(a[g][c] - d[g][c]) for g in st.GROUPS for c in st.cells())
    assert moved == 4 * len(st.DESIGN_SWAPS)


def test_design_is_balanced():
    alloc = st.allocation()
    assert {g: sum(a.values()) for g, a in alloc.items()} == st.COUNTS
    for c in st.cells():
        assert sum(alloc[g][c] for g in st.GROUPS) == (st.SINGLES if len(c) == 1 else st.PAIRS)
        for g in st.GROUPS:
            share = st.COUNTS[g] * (st.SINGLES if len(c) == 1 else st.PAIRS) / 100
            assert int(share) <= alloc[g][c] <= int(share) + 1
    slots = [c for g in st.GROUPS for c in st.design()[g]]
    assert len(slots) == 100 and all(1 <= len(c) <= 2 for c in slots)
    for m in st.COMPLICATIONS:
        assert 40 <= sum(m in c for c in slots) <= 50
    for a, b in itertools.combinations(st.COMPLICATIONS, 2):
        assert sum(a in c and b in c for c in slots) >= 10
    assert st.design() == st.design()


def test_generator_is_deterministic_and_derives_its_stratum():
    for gi, g in enumerate(st.GROUPS):
        for j, combo in enumerate(st.design()[g][:3]):
            seed = 98_000_000 + 10_000 * gi + j
            a, b = st.generate("x", seed, g, j, combo), st.generate("x", seed, g, j, combo)
            assert a.params == b.params
            assert sorted(st.complications(a.params)) == sorted(combo)
            d = st.derive(a.scenario())
            assert d["variant"] == st.VARIANT[g] and d["expected"] == st.EXPECTED[g]
            assert not st.timing_problems(a.params) and not st.scope_problems(a.params)


# --- the built manifest --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def manifest():
    return json.loads(st.MANIFEST.read_text())


@needs_manifest
def test_manifest_mix_seeds_and_design(manifest):
    es = manifest["entries"]
    assert manifest["split"] == st.SPLIT and len(es) == 100
    assert collections.Counter(e["group"] for e in es) == collections.Counter(st.COUNTS)
    assert all(e["variant"] == st.VARIANT[e["group"]] and e["expected"] == st.EXPECTED[e["group"]] for e in es)
    assert all(e["seed"] == 11_800_000 + 10_000 * st.GROUPS.index(e["group"]) + e["k"] for e in es)
    assert len({e["seed"] for e in es}) == 100
    for m in st.COMPLICATIONS:
        assert 40 <= sum(m in e["complications"] for e in es) <= 50
    for a, b in itertools.combinations(st.COMPLICATIONS, 2):
        assert sum(a in e["complications"] and b in e["complications"] for e in es) >= 10
    for g in st.GROUPS:
        cells = collections.Counter(tuple(e["combo"]) for e in es if e["group"] == g)
        assert cells == collections.Counter(st.design()[g])
    assert [e["position"] for e in es] == list(range(100))


@needs_manifest
def test_manifest_content_and_labels(manifest):
    fs = st.load(st.MANIFEST)
    assert list(fs) == [e["case_id"] for e in manifest["entries"]]
    for e in manifest["entries"]:
        p = e["params"]
        assert st.params_sha(p) == e["params_sha256"]
        assert sorted(st.complications(p)) == sorted(e["complications"]) == sorted(e["combo"])
        assert not st.timing_problems(p) and not st.scope_problems(p) and not st.admissibility_problems(p)
        assert all(t < 0 for _, t in st.timed_view(p)["receipts"]) and len(st.timed_view(p)["receipts"]) <= 5
        if st.is_timed(p):
            assert tuple(sorted(p)) == tuple(sorted(st.TIMED_KEYS))
            assert {"M2", "M3"} & set(e["complications"]) or len(p["timed_receipts"]) >= 4   # fixtures.build cannot
        else:
            assert tuple(sorted(p)) == tuple(sorted(st.FROZEN_KEYS)) and len(p["receipts"]) <= 3
            assert "M3" not in e["complications"]
        s = fs[e["case_id"]]()
        d = st.derive(s)
        assert (d["variant"], d["expected"], d["discrepancy_type"]) == (e["variant"], e["expected"],
                                                                        e["discrepancy_type"])
        assert st.param_keys(e["seed"], p) == e["keys"] and st.visible_keys(e["seed"], s) == e["visible_keys"]


@needs_manifest
def test_reference_agent_solves_every_case_under_g7_and_g8(manifest):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "stress"))
    from common import reference_episode
    fs = st.load(st.MANIFEST)
    for cid, f in fs.items():
        r = reference_episode(cid, f)
        assert r["error"] is None and r["g7"]["reward"] == 1 and r["g8"]["reward"] == 1, (cid, r["g8"])


@needs_manifest
def test_no_seed_or_key_shared_with_any_split(manifest):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = [DIR / f"{s}.json" for s in ("sft", "rl", "dev", "regression", "test", "test_reserve", "v6",
                                         "policy_shift")] + [root / "manifests/forgetting_l1/forgetting_l1.json"]
    seeds, ks = set(), set()
    for p in paths:
        for e in json.loads(p.read_text())["entries"]:   # the reserve's keys by program; nothing is printed
            seeds.add(e["seed"])
            ks |= set(keys(e["seed"], _tuples(e["params"])).values())
    es = manifest["entries"]
    assert not seeds & {e["seed"] for e in es}
    assert not ks & {k for e in es for k in e["keys"].values()}
    # also visible keys and format-independent figures, against every split and the earlier sources (v1 sets, fresh_1000, its regeneration, dev_cases, boundary)
    assert {e["case_id"]: st.leakage_matches(e["seed"], e["params"]) for e in es
            if st.leakage_matches(e["seed"], e["params"])} == {}
    assert {"sft", "test_reserve", "dev_cases", "boundary", "fresh_1000_regenerated"} <= set(
        st.existing_index()["cases_by_source"])


@needs_manifest
def test_loader_refuses_a_timed_reencoding_of_an_existing_case(manifest, tmp_path):
    """A frozen sft case written in the timed format has new parameter keys but the same visible content and figures:
    the loader refuses it."""
    sft = next(e for e in json.loads((DIR / "sft.json").read_text())["entries"] if e["case_id"] == "sft-V1-1")
    timed = _timed(_tuples(sft["params"]))
    assert keys(sft["seed"], timed) != keys(sft["seed"], _tuples(sft["params"]))
    assert st.leakage_matches(sft["seed"] + 1, timed)       # another seed too: visible near/group and figures match
    doc = json.loads(json.dumps(manifest))
    e = doc["entries"][0]
    e["params"], e["params_sha256"] = timed, st.params_sha(timed)
    p = tmp_path / "messiness.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(st.StressRefused, match="matches an existing case"):
        st.load(p)


@needs_manifest
def test_loader_refuses_wrong_stored_labels(manifest, tmp_path):
    for field, value in (("expected", "pending"), ("variant", "V4_pending"), ("discrepancy_type", "none")):
        doc = json.loads(json.dumps(manifest))
        e = next(x for x in doc["entries"] if x[field] != value)
        e[field] = value
        p = tmp_path / "messiness.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(st.StressRefused, match=f"stored {field}"):
            st.load(p)


@needs_manifest
def test_every_m3_case_is_a_true_reissue(manifest):
    """Set on 2026-11-06: each M3 re-issue is posted after its original's cancellation (an active invoice of the
    cancelled one's quantity); no case has a same-quantity active invoice posted while the original was still
    active."""
    for e in manifest["entries"]:
        o = st.timed_view(e["params"])["other_invoices"]
        pairs = [(pa, cc) for qa, pa, ca in o if ca is None for qc, pc, cc in o
                 if cc is not None and qc == qa and pa > pc]
        assert all(pa > cc for pa, cc in pairs), e["case_id"]
        assert bool(pairs) == ("M3" in e["complications"]), e["case_id"]


@needs_manifest
def test_pages_at_reset_mirror_the_parameters(manifest):
    """For every case, both formats, the pages at reset show
    exactly the parameters: the PO line (quantity and price), the invoice line, the receipts with their quantities, the
    other invoices, the Event history, and the blocking comparisons, recomputed here from the parameters at the block
    (POLICY A3 to A6) without the stress module's derive()."""
    from decimal import Decimal
    from ap_sim.events.clock import format_time
    from ap_sim.interfaces.structured import StructuredSurface
    from ap_sim.interfaces.tools import AgentTools, ToolLayer
    fs = st.load(st.MANIFEST)
    for e in manifest["entries"]:
        p, v = e["params"], st.timed_view(e["params"])
        tools = AgentTools(ToolLayer(fs[e["case_id"]](), StructuredSurface(), "surfaces/4"))
        obs = tools.initial_observation

        def go(name):
            nonlocal obs
            cid = next(c["id"] for g in obs["regions"] for c in g["controls"] if c["name"] == name)
            obs = tools.click(cid)["observation"]
            return [r for g in obs["regions"] if g["role"] not in ("actions", "navigation") for r in g["records"]]

        inv = go("Invoices on PO line")
        shown = sorted((r["fields"]["state"], r["fields"]["quantity"]["value"], r["fields"]["posted_at"],
                        r["fields"]["cancelled_at"]) for r in inv)
        want = sorted([("cancelled" if c is not None else "posted_released", str(q), format_time(p),
                        None if c is None else format_time(c)) for q, p, c in v["other_invoices"]]
                      + [("posted_blocked", str(v["invoice_qty"]), format_time(v["blocked_at"]), None)])
        assert shown == want, e["case_id"]
        events = sorted((r["fields"]["time"], r["fields"]["type"]) for r in go("Event history"))
        want = sorted([(format_time(t), "receipt_posted") for _, t in v["receipts"]]
                      + [(format_time(p), "invoice_posted") for _, p, _ in v["other_invoices"]]
                      + [(format_time(c), "invoice_cancelled") for _, _, c in v["other_invoices"] if c is not None]
                      + [(format_time(v["blocked_at"]), "invoice_posted"),
                         (format_time(v["blocked_at"]), "invoice_checked")])
        assert events == want, e["case_id"]
        receipts = sorted((r["fields"]["posted_at"], r["fields"]["quantity"]["value"])
                          for r in go("Receipt history") if "quantity" in r["fields"] and "posted_at" in r["fields"])
        assert receipts == sorted((format_time(t), str(q)) for q, t in v["receipts"]), e["case_id"]
        po = [r["fields"] for r in go("Purchase order") if "ordered_quantity" in r["fields"]]
        assert len(po) == 1, e["case_id"]
        assert (po[0]["ordered_quantity"]["value"], po[0]["unit_price"]["value"]) == (str(p["po_qty"]), p["price"]), \
            e["case_id"]
        first = tools.initial_observation
        line = [r["fields"] for g in first["regions"] if g["role"] == "invoice_lines" for r in g["records"]]
        assert (line[0]["quantity"]["value"], line[0]["unit_price"]["value"]) == (str(p["invoice_qty"]), p["price"]), \
            e["case_id"]
        shown = {r["fields"]["comparison"]: r["fields"]["result"]
                 for g in first["regions"] if g["role"] == "blocking_comparisons" for r in g["records"]}
        b, price, inv = v["blocked_at"], Decimal(p["price"]), p["invoice_qty"]
        net = sum(q for q, t in v["receipts"] if t <= b)
        active = [q for q, pt, c in v["other_invoices"] if pt <= b and (c is None or c > b)]
        po_amount = (p["po_qty"] * price).quantize(Decimal("0.01"))
        amount = sum((q * price).quantize(Decimal("0.01")) for q in active + [inv])
        want = {"invoice_vs_receipt": inv > net - sum(active),
                "invoice_vs_po_quantity": sum(active) + inv > p["po_qty"] * Decimal("1.10"),
                "invoice_vs_po_amount": amount > min(po_amount * Decimal("1.10"), po_amount + 1000),
                "receipt_vs_po": net > p["po_qty"] * Decimal("1.05"), "price": False}
        assert shown == {k: "fail" if f else "pass" for k, f in want.items()}, e["case_id"]


@needs_manifest
def test_loader_refuses_tampering(manifest, tmp_path):
    doc = json.loads(json.dumps(manifest))
    doc["entries"][0]["params"]["invoice_qty"] += 1
    p = tmp_path / "messiness.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(st.StressRefused):
        st.load(p)
    doc = json.loads(json.dumps(manifest))
    doc["split"] = "test"
    p.write_text(json.dumps(doc))
    with pytest.raises(st.StressRefused):
        st.load(p)
    with pytest.raises(st.StressRefused):
        st.load(st.MANIFEST, require_frozen=True) if not (st.MANIFEST.parent / "FREEZE.json").exists() else \
            st.load(p, require_frozen=True)

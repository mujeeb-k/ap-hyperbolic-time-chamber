"""Build the v2 splits: generate, deduplicate, split and order the v2 cases, and write the manifests to manifests/v2/.

Every rule here was fixed on 2026-10-30, before any case existed:

- Seeds: 11,000,000 + 100,000 x split index + 10,000 x group index + k. Policy shift: pairs from seed 117000,
  boundary from seed 118000 (policy_shift's own seed x 100 + k scheme).
- Forced variables from random.Random(f"v2-phase2-strata:{seed}"); cells cycle within each group by the accepted
  index j (so a redraw never changes the mix).
- A case is redrawn (next k) if its exact, near or group key matches an earlier case (the 1,580, the 30 sweep cases,
  the 55 dev cases, the earlier boundary cases) or a new case generated before it. Splits are generated in priority
  order: test, test reserve, V6, policy shift, dev, regression, RL, SFT.
- Order: a stratified interleave (each group's cases spread evenly), seeds 20261103 (test, then reserve, one stream)
  and 20261031 (RL). RL 0-79 pilot, 80-479 main. Profiling sample: 250 of the 480, stratified, seed 20261102.

No model is run. Labels (expected outcome, discrepancy_type) come from grader g7 on an unacted episode log.

    .venv/bin/python scripts/phase2/build_splits.py [--out manifests/v2]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

from ap_sim.generator import causal, ladder, policy_shift   # noqa: E402
from ap_sim.generator.fixtures import DEV_CASES   # noqa: E402
from ap_sim.generator.manifests import _tuples, params_sha   # noqa: E402
from ap_sim.generator.splits import gap_at_start, has_reversal, invoice_amount, is_v6, keys   # noqa: E402
from ap_sim.grader.grader import GRADER_VERSION, grade   # noqa: E402
from ap_sim.interfaces.structured import StructuredSurface   # noqa: E402
from ap_sim.interfaces.tools import ToolLayer   # noqa: E402

SEED_BASE = 11_000_000
SPLIT_INDEX = {"test": 0, "test_reserve": 1, "v6": 2, "dev": 3, "regression": 4, "rl": 5, "sft": 6}
SIZES = {"test": 200, "test_reserve": 100, "v6": 100, "dev": 60, "regression": 40, "rl": 480, "sft": 300}
PRIORITY = ("test", "test_reserve", "v6", "policy_shift", "dev", "regression", "rl", "sft")
GROUPS = ("V1", "V2", "V3a", "V3b", "V4", "V5")
SHARES = {"V1": 20, "V2": 18, "V3a": 10, "V3b": 10, "V4": 22, "V5": 20}   # percent
V6_COUNTS = {"V1": 30, "V2": 20, "V3a": 10, "V3b": 10, "V4": 30}
VARIANT = {"V1": "V1_delivered_unrecorded", "V2": "V2_not_delivered", "V3a": "V3a_late_delivered",
           "V3b": "V3b_late_not_delivered", "V4": "V4_pending", "V5": "V5_stale_covered"}
PS_SEED, PB_SEED, PS_PAIRS = 117000, 118000, 20
ORDER_SEED = {"test": 20261103, "rl": 20261031}
PROFILING_SEED, PROFILING_N = 20261102, 250
PILOT = 80


def counts(n: int) -> dict[str, int]:
    """Largest-remainder rounding of the group shares (ties in group order)."""
    raw = {g: n * SHARES[g] / 100 for g in GROUPS}
    out = {g: int(v) for g, v in raw.items()}
    for g in sorted(GROUPS, key=lambda g: (-(raw[g] - out[g]), GROUPS.index(g)))[:n - sum(out.values())]:
        out[g] += 1
    return out


def force(group: str, j: int, seed: int, v6: bool) -> dict:
    """The forced variables of the j-th accepted case of a group (cells cycle by j)."""
    rng = random.Random(f"v2-phase2-strata:{seed}")
    f = {"reversal": v6}
    if group == "V5":
        return f | ({"gap": 0, "stale_receipt": True, "n_receipts": rng.choice([2, 2, 3])} if j % 2 == 0 else
                    {"gap": 0, "stale_receipt": False, "cancellation": "after_block"})
    f["gap"] = 1 if j % 10 == 9 else rng.randint(2, 25)
    if group == "V1":
        f |= {"delay": ("early", "early", "mid", "mid")[j % 4], "unrecorded": ("equal", "above")[j % 2]}
    elif group == "V2":
        f |= {"delay": ("early", "mid")[j % 2], "unrecorded": "zero"}
    elif group == "V3a":
        f |= {"delay": "late", "unrecorded": ("equal", "above")[j % 2]}
    elif group == "V3b":
        f |= {"delay": "late", "unrecorded": "zero"}
    elif group == "V4":
        f |= {"delay": ("beyond", "never")[j % 2], "unrecorded": rng.choice(["zero", "equal", "above"])}
    return f


# --- the earlier cases ---------------------------------------------------------------------------------------------
def earlier_keys() -> tuple[dict[str, str], dict]:
    """Every key of the cases used before the v2 splits were built -> where it came from. fresh_1000.json is written by
    scripts/phase0_5_compare.py."""
    out, n = {}, collections.Counter()

    def add(source, seed, params):
        for k in keys(seed, _tuples(params)).values():
            out.setdefault(k, source)
        n[source.split("/")[0]] += 1

    for f in sorted((ROOT / "manifests").glob("frozen*/*.json")):
        if f.name == "FREEZE.json":
            continue
        for e in json.loads(f.read_text())["entries"]:
            add(f"frozen:{f.stem}/{e['case_id']}", e["seed"], e["params"])
    fresh = ROOT / "runs" / "phase0_5" / "fresh_1000.json"
    for e in json.loads(fresh.read_text())["entries"]:
        add(f"fresh_1000/{e['case_id']}", e["seed"], e["params"])
    for seed in range(9_100_000, 9_101_000):   # the same seeds regenerated now (63 differ: the cancellation fix)
        c = causal.generate(f"F-{seed}", seed)
        add(f"fresh_1000_regenerated/{c.case_id}", seed, c.params)
    for e in json.loads((ROOT / "manifests" / "dev_cases.json").read_text())["cases"]:
        add(f"dev_cases/{e['id']}", e["seed"], e["params"])
    for seed in (4242, 5100, 5200, 5300):
        for c in causal.boundary_cases(seed):
            add(f"boundary/{c.case_id}", c.seed, c.params)
    assert len(DEV_CASES) == 55
    return out, dict(n)


# --- generation ----------------------------------------------------------------------------------------------------
class Pool:
    """The keys in use, and the collision tally per split."""

    def __init__(self, earlier: dict[str, str]):
        self.earlier, self.new = earlier, {}
        self.collisions = collections.defaultdict(collections.Counter)

    def clash(self, split: str, ks: dict) -> bool:
        for kind in ("exact", "near", "group"):
            if ks[kind] in self.earlier:
                self.collisions[split][f"{kind} key, earlier case"] += 1
                return True
            if ks[kind] in self.new:
                self.collisions[split][f"{kind} key, new case"] += 1
                return True
        return False

    def take(self, ks: dict, owner: str) -> None:
        for k in ks.values():
            self.new[k] = owner


def label(c: causal.Case) -> dict:
    s = c.scenario()
    v = grade(s, ToolLayer(s, StructuredSurface(), "surfaces/4").episode_log())
    return {"expected": v.expected, "discrepancy_type": v.diagnostics["discrepancy_type"]["required"],
            **invoice_amount(s)}


def entry(split: str, c: causal.Case, group: str, f: dict | None) -> dict:
    params = {k: (list(map(list, v)) if k == "reversals" else list(v) if isinstance(v, tuple) else v)
              for k, v in c.params.items()}
    return {"case_id": c.case_id, "split": split, "seed": c.seed, "force": f, "generator": causal.GENERATOR_VERSION,
            "policy": c.params.get("policy", "sandbox-v1"), "group": group, "variant": c.variant,
            "signature": list(c.signature), "level": ladder.level(c), "delay": c.params["delay"], **label(c),
            "keys": keys(c.seed, c.params), "params": params, "params_sha256": params_sha(_tuples(params))}


def generate_split(split: str, pool: Pool) -> list[dict]:
    v6 = split == "v6"
    want = V6_COUNTS if v6 else counts(SIZES[split])
    out = []
    for gi, group in enumerate(GROUPS):
        k = j = 0
        while j < want.get(group, 0):
            seed = SEED_BASE + 100_000 * SPLIT_INDEX[split] + 10_000 * gi + k
            k += 1
            f = force(group, j, seed, v6)
            c = causal.generate(f"{split}-{group}-{seed - SEED_BASE - 100_000 * SPLIT_INDEX[split] - 10_000 * gi}",
                                seed, f)
            want_variant = ("V6_reversal+" if v6 else "") + VARIANT[group]
            if c.variant != want_variant:
                raise AssertionError(f"{c.case_id}: forced {f} gave {c.variant}, not {want_variant}")
            if is_v6(c.params) != v6 or has_reversal(c.params) != v6:
                raise AssertionError(f"{c.case_id}: V6 content check disagrees with the forcing")
            ks = keys(c.seed, c.params)
            if pool.clash(split, ks):
                continue
            pool.take(ks, f"{split}/{c.case_id}")
            out.append(entry(split, c, group, f))
            j += 1
    return out


def generate_policy_shift(pool: Pool) -> list[dict]:
    """20 flipping pairs (v1 keep_blocked, v2 release) and the at-limit / one-over boundary pair, as in v1's slice."""
    out, s = [], PS_SEED
    while len(out) < 2 * PS_PAIRS:
        p = policy_shift.pair(s)
        s += 1
        if not p or policy_shift.expected(p[0]) != "keep_blocked" or policy_shift.expected(p[1]) != "release":
            continue
        ks = keys(p[0].seed, p[0].params)   # the v2 copy shares the group key by design
        kv2 = keys(p[1].seed, p[1].params)
        if pool.clash("policy_shift", ks) or pool.clash("policy_shift", {"exact": kv2["exact"], "near": kv2["near"],
                                                                            "group": "-"}):
            continue
        pool.take(ks, f"policy_shift/{p[0].case_id}")
        pool.take(kv2, f"policy_shift/{p[1].case_id}")
        out += [entry("policy_shift", c, "PS", None) for c in p]
    b, seed = None, PB_SEED
    while b is None:
        b = policy_shift.boundary(seed)
        groups = {keys(x.seed, x.params)["group"] for v1, v2, _ in b for x in (v1, v2)}
        clash = any(pool.clash("policy_shift", {"exact": "-", "near": "-", "group": g}) for g in groups)
        clash = clash or any(pool.clash("policy_shift", {**keys(x.seed, x.params), "group": "-"})
                             for v1, v2, _ in b for x in (v1, v2))
        if clash:
            b, seed = None, seed + 1
    for v1, v2, _ in b:
        for x in (v1, v2):
            pool.take(keys(x.seed, x.params), f"policy_shift/{x.case_id}")
            out.append(entry("policy_shift", x, "PB", None))
    return out


# --- order and profiling sample ------------------------------------------------------------------------------------
def interleave(entries: list[dict], rng: random.Random) -> list[dict]:
    """Each group's cases (shuffled) spread evenly through the order: case i of a group of n gets (i + u) / n."""
    keyed = []
    for gi, group in enumerate(GROUPS):
        members = [e for e in entries if e["group"] == group]
        rng.shuffle(members)
        u = rng.random()
        keyed += [((i + u) / len(members), gi, e) for i, e in enumerate(members)]
    return [e for *_, e in sorted(keyed, key=lambda t: (t[0], t[1]))]


def profiling_sample(rl: list[dict]) -> list[str]:
    rng = random.Random(PROFILING_SEED)
    by = {g: sorted(e["case_id"] for e in rl if e["group"] == g) for g in GROUPS}
    raw = {g: PROFILING_N * len(by[g]) / len(rl) for g in GROUPS}
    n = {g: int(v) for g, v in raw.items()}
    for g in sorted(GROUPS, key=lambda g: (-(raw[g] - n[g]), GROUPS.index(g)))[:PROFILING_N - sum(n.values())]:
        n[g] += 1
    return sorted(cid for g in GROUPS for cid in rng.sample(by[g], n[g]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "manifests" / "v2"))
    ap.add_argument("--dry-run", action="store_true", help="allow uncommitted src/ (output is not for freezing)")
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=ROOT).stdout
    if dirty and not a.dry_run:
        raise SystemExit("src/ has uncommitted changes: generate from a committed generator")
    earlier, earlier_counts = earlier_keys()
    pool = Pool(earlier)
    splits = {}
    for split in PRIORITY:
        splits[split] = generate_policy_shift(pool) if split == "policy_shift" else generate_split(split, pool)
        print(split, len(splits[split]), dict(pool.collisions[split]), flush=True)
    rng = random.Random(ORDER_SEED["test"])
    splits["test"] = interleave(splits["test"], rng)
    splits["test_reserve"] = interleave(splits["test_reserve"], rng)
    splits["rl"] = interleave(splits["rl"], random.Random(ORDER_SEED["rl"]))
    for i, e in enumerate(splits["rl"]):
        e["rl_role"] = "pilot" if i < PILOT else "main"
        e["rl_step"] = i // 8 if i < PILOT else (i - PILOT) // 8
    prof = profiling_sample(splits["rl"])
    for e in splits["rl"]:
        e["profiling_sample"] = e["case_id"] in prof
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        for i, e in enumerate(entries):
            e["position"] = i
        doc = {"split": split, "design": "docs/V2.md, Phase 2 approved (2026-10-30)", "generator": causal.GENERATOR_VERSION,
               "random_stream": causal.RNG_STREAM, "generator_commit": commit, "grader": GRADER_VERSION,
               "seed_base": None if split == "policy_shift" else SEED_BASE + 100_000 * SPLIT_INDEX[split],
               "order_seed": ORDER_SEED.get("test" if split.startswith("test") else split),
               "collisions_redrawn": dict(pool.collisions[split]), "cases": len(entries), "entries": entries}
        if split == "policy_shift":
            doc["seeds"] = {"pairs": PS_SEED, "boundary": PB_SEED}
        if split == "rl":
            doc["profiling"] = {"seed": PROFILING_SEED, "n": PROFILING_N, "case_ids": prof}
            doc["pilot"] = PILOT
        (out / f"{split}.json").write_text(json.dumps(doc, indent=1) + "\n")
    summary = {"generator_commit": commit, "earlier_cases_by_source": earlier_counts,
               "collisions_redrawn": {s: dict(pool.collisions[s]) for s in PRIORITY},
               "sizes": {s: len(e) for s, e in splits.items()},
               "mix": {s: {"group": dict(collections.Counter(e["group"] for e in es)),
                           "expected": dict(collections.Counter(e["expected"] for e in es)),
                           "discrepancy_type": dict(collections.Counter(e["discrepancy_type"] for e in es)),
                           "policy": dict(collections.Counter(e["policy"] for e in es)),
                           "delay": dict(collections.Counter(str(e["delay"]) for e in es)),
                           "gap_1": sum(gap_at_start(e["params"]) == 1 for e in es)}
                       for s, es in splits.items()}}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

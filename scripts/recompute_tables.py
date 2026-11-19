"""Recompute the paper's results tables, paired intervals and McNemar tests from data/outcomes.csv and the frozen manifests, and check each value
against the paper. CPU only. No transcript, model or network is needed.

    python scripts/recompute_tables.py [--csv data/outcomes.csv] [--show]

The paper's tables are copied below (PAPER_TABLES) exactly as printed in the draft they were taken from, under that draft's numbers and its repository
names (Table S1). Their values equal the final paper's. The checks print the final paper's numbers: draft Tables 1 and S20 are Table 1, S6–S12 are
S9–S15, S14 is S17 (and draft Table 1's known false passes), and Figure 1 is Figure 2. Each cell the CSV supports is recomputed and compared with
the printed cell.
Cells the CSV does not support are listed as skipped, with the reason. The printed intervals and p-values in the text are checked the same way
(TEXT_CHECKS). Paired intervals use the frozen paired bootstrap (src/ap_sim/evaluation/metrics.py, 10,000 resamples, seed 20261101). McNemar
p-values are exact and two-sided, as in scripts/posthoc/manuscript_counts.py. The unseen-signature subset and the stress groups are read from
the frozen manifests with the paper's own code (scripts/gpu/final_analysis.py, scripts/gpu/followup_analysis.py). Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re
import sys

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/gpu")]

from ap_sim.evaluation import metrics   # noqa: E402

SETS = {"test": "test", "reserve": "test_reserve", "stress": "stress", "v6": "v6", "policy shift": "policy_shift", "forgetting": "forgetting_l1"}
PRIMARY_ORDER = ("wrong_decision", "cancelled_invoice_omission", "missing_evidence", "derived_figure", "other")   # manuscript_counts.PRIMARY


def mcnemar_exact(gained: int, lost: int) -> float:
    """Exact two-sided McNemar p-value, as scripts/posthoc/manuscript_counts.mcnemar_exact."""
    n, k = gained + lost, min(gained, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def primary(labels: set[str]) -> str:
    """Table S11's one category per failure, in PRIMARY_ORDER (manuscript_counts.primary)."""
    return next((k for k in PRIMARY_ORDER[:-1] if k in labels), "other")


class Data:
    def __init__(self, path: pathlib.Path):
        self.rows = list(csv.DictReader(path.open()))
        self.runs: dict[tuple[str, str, str], dict[str, dict]] = collections.defaultdict(dict)
        for r in self.rows:
            self.runs[(r["model"], r["evaluation"], r["set"])][r["case_id"]] = r

    def run(self, model: str, evaluation: str, the_set: str) -> dict[str, dict]:
        got = self.runs.get((model, evaluation, the_set))
        if not got:
            raise KeyError(f"no rows for {model} / {evaluation} / {the_set}")
        return got

    def success(self, *key) -> dict[str, float]:
        return {c: float(r["full_success"] == "true") for c, r in self.run(*key).items()}

    def records(self, *key) -> list[dict]:
        """Minimal records for the paper's analysis functions: case and success only."""
        return [{"case": c, "success": r["full_success"] == "true"} for c, r in self.run(*key).items()]


def who(label: str, the_set: str) -> tuple[str, str]:
    """A paper row label and set -> (model, evaluation) in the CSV."""
    label = re.sub(r"^\*Reference:\* ", "", label).strip()
    label = label[0].upper() + label[1:]
    if label.startswith("Claude Sonnet 5.5"):
        return "Claude Sonnet 5.5", "reference"
    if label.startswith("SFT-v2, re-evaluated"):
        return "SFT-v2", "controls"
    if label.startswith("SFT-v2 + prompt reminder"):
        return "SFT-v2 + prompt reminder", "controls"
    if label.startswith("Second SFT run"):
        return "Second SFT run", "controls"
    if label.startswith("Base"):
        return "Base", "primary"
    later = the_set in ("test_reserve", "stress")
    for name in ("GRPO seed 2", "GRPO seed 3"):
        if label.startswith(name):
            return name, "follow-up"
    for name in ("GRPO seed 1", "SFT-v2", "RFT"):
        if label.startswith(name):
            return name, "follow-up" if later else "primary"
    raise KeyError(label)


def pp(x: float, test: bool) -> str:
    v = round(100 * x, 1)
    if v == 0:
        return "0.0" if test else "0"
    return f"{v:+.1f}" if test else f"{v:+.0f}"


def interval(d: dict, test: bool) -> str:
    return f"{pp(d['difference'], test)} [{pp(d['low'], test)}, {pp(d['high'], test)}]"


def norm(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").strip()


def tables(text: str) -> dict[str, list[list[str]]]:
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"\*\*Table (S?\d+)\.\*\*", line)
        if m:
            name, out[m.group(1)] = m.group(1), []
        elif name and line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            out[name].append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


class Checker:
    def __init__(self):
        self.ok, self.bad, self.skipped = 0, [], collections.Counter()

    def cell(self, where: str, got, printed: str):
        if norm(str(got)) == norm(printed):
            self.ok += 1
        else:
            self.bad.append(f"{where}: recomputed {got!r}, printed {printed!r}")

    def skip(self, why: str, n: int = 1):
        self.skipped[why] += n


def check_tables(d: Data, ck: Checker, show: bool):
    t = tables(PAPER_TABLES)

    # Table 1: right final action, success, heuristic right final action; known false passes are Table S17
    the_set = None
    for row in t["1"][1:]:
        label = row[0]
        m = re.match(r"\*(Test|V6|Policy shift|Stress) \((\d+) cases\)\*", label)
        if m:
            the_set = SETS[m.group(1).lower()]
            h = d.run("Scripted heuristic", "post hoc", the_set)
            ck.cell(f"Table 1 heuristic {the_set}", sum(r["right_final_action"] == "true" for r in h.values()), row[4])
            continue
        if label.startswith("*Exploratory"):
            continue
        run = d.run(*who(label, the_set), the_set)
        ck.cell(f"Table 1 {the_set} {label} right final action", sum(r["right_final_action"] == "true" for r in run.values()), row[1])
        ck.cell(f"Table 1 {the_set} {label} success", sum(r["full_success"] == "true" for r in run.values()), row[2])
        ck.cell(f"Table S17 {the_set} {label} known false passes", sum(r["known_false_pass"] == "true" for r in run.values()), row[3])

    # Table S9: success and right final action ("a / b / c" by the row's sets); report valid and the base's outcome-correct counts are not in the CSV
    for row in t["S6"][1:]:
        sets = [SETS[s.strip().lower()] for s in row[1].split("/")]
        for col, field in ((2, "full_success"), (3, "right_final_action")):
            printed = re.sub(r"\s*\(.*\)$", "", row[col]).split("/")
            for s, p in zip(sets, printed):
                run = d.run(*who(row[0], s), s)
                ck.cell(f"Table S9 {row[0]} {s} {field}", sum(r[field] == "true" for r in run.values()), p)
        ck.skip("Table S9 report valid: needs the record's finish_accepted, not in the CSV", len(sets))
        if "outcome correct" in row[3]:
            ck.skip("Table S9 base outcome-correct counts: need the record's outcome_correct, not in the CSV", len(sets))

    # Table S10: test success by family (from the case id), sampled success, pass^8, sampled success on V4 (primary evaluation)
    head = t["S7"][0]
    for row in t["S7"][1:]:
        run = d.run(*who(row[0], "test"), "test")
        for i, h in enumerate(head[1:7], 1):
            fam = h.split()[0]
            ck.cell(f"Table S10 {row[0]} {fam}", sum(r["full_success"] == "true" for c, r in run.items() if c.split("-")[1] == fam), row[i])
        k = {c: r["sampled_successes_of_8"] for c, r in run.items()}
        if all(v == "" for v in k.values()):
            for i in (7, 8, 9):
                ck.cell(f"Table S10 {row[0]} {head[i]} (no samples)", "–", row[i])
            continue
        k = {c: int(v) for c, v in k.items()}
        ck.cell(f"Table S10 {row[0]} sampled success", f"{sum(k.values()) / (8 * len(k)):.3f}", row[7])
        p8 = sum(v == 8 for v in k.values()) / len(k)
        ck.cell(f"Table S10 {row[0]} pass^8", f"{p8:.3f}".rstrip("0"), row[8])
        v4 = [v for c, v in k.items() if c.split("-")[1] == "V4"]
        ck.cell(f"Table S10 {row[0]} sampled success on V4", f"{sum(v4) / (8 * len(v4)):.3f}", row[9])

    # Tables S11 and S12: stress failures by primary category and by every label (paper_labels)
    for name, final, cols in (("S8", "S11", {2: "wrong_decision", 3: "cancelled_invoice_omission", 4: "derived_figure", 5: "other"}),
                       ("S9", "S12", {2: "wrong_decision", 3: "missing_evidence", 4: "derived_figure", 5: "other_citation_problem", 6: "discrepancy_type_wrong", 7: "no_flag"})):
        for row in t[name][1:]:
            run = d.run(*who(row[0], "stress"), "stress")
            failures = [set(r["paper_labels"].split(";")) for r in run.values() if r["full_success"] == "false"]
            ck.cell(f"Table {final} {row[0]} failures", len(failures), row[1])
            if name == "S8":
                counts = collections.Counter(primary(f) for f in failures)
                if counts["missing_evidence"]:
                    ck.bad.append(f"Table S11 {row[0]}: {counts['missing_evidence']} failures fall in 'other missing evidence', which the table has no column for")
            else:
                counts = collections.Counter(x for f in failures for x in f)
                if any(("missing_evidence" in f) != ("cancelled_invoice_omission" in f) for f in failures):
                    ck.bad.append(f"Table S12 {row[0]}: missing evidence is not all canceled-invoice")
            for i, lab in cols.items():
                ck.cell(f"Table {final} {row[0]} {lab}", counts[lab], row[i])

    # Table S13: grader success and variant success; the (a)/(b)/(o) entry counts are per derived entry, not per case
    for row in t["S10"][1:]:
        s, label = [x.strip() for x in row[0].split(",", 1)]
        the_set = SETS[s.lower()]
        run = d.run(*who(label, the_set), the_set)
        ck.cell(f"Table S13 {row[0]} grader success", sum(r["full_success"] == "true" for r in run.values()), row[1])
        ck.cell(f"Table S13 {row[0]} variant success", sum(r["variant_success"] == "true" for r in run.values()), row[2])
        ck.skip("Table S13 (a)/(b)/(o): counts of derived entries, not per-case values")

    # Table S14: other paired comparisons (percentage points, 95% paired-bootstrap intervals)
    pairs = {"GRPO seed 1 − RFT": ("GRPO seed 1", "RFT"), "GRPO seed 2 − RFT": ("GRPO seed 2", "RFT"), "GRPO seed 3 − RFT": ("GRPO seed 3", "RFT"),
             "Second SFT run − SFT-v2": ("Second SFT run", "SFT-v2"), "SFT-v2 + prompt reminder − SFT-v2": ("SFT-v2 + prompt reminder", "SFT-v2"),
             "Second SFT run − (SFT-v2 + prompt reminder)": ("Second SFT run", "SFT-v2 + prompt reminder")}
    for row in t["S11"][1:]:
        a, b = pairs[row[0]]
        for i, s in ((1, "test"), (2, "test_reserve"), (3, "stress")):
            if row[i] == "–":
                continue
            x, y = d.success(*who(a, s), s), d.success(*who(b, s), s)
            ck.cell(f"Table S14 {row[0]} {s}", interval(metrics.paired_bootstrap(y, x), s == "test"), row[i])

    # Table S15: test episodes with an attempted unsafe action and the attempts; decision-error columns need final actions, not in the CSV
    for row in t["S12"][1:]:
        run = d.run(*who(row[0], "test"), "test")
        n = [int(r["attempted_unsafe"]) for r in run.values()]
        eps, att = sum(x > 0 for x in n), sum(n)
        got = f"{eps} ({att}, 0)" if eps else "0"
        printed = row[3]
        ck.cell(f"Table S15 {row[0]} attempted unsafe (episodes, attempts)", re.sub(r", 0\)$", ")", got), re.sub(r", 0\)$", ")", printed))
        ck.skip("Table S15 executed unsafe actions: not in the CSV (zero by construction)", 1 if eps else 0)
        ck.skip("Table S15 incorrect hold / no counted final action, and weighted error: need final actions, stop reasons and resolution times", 3)

    # Table S17: the false passes the audit detected, with the cell's successes
    cells = {"SFT-v2, V6 (primary evaluation)": ("SFT-v2", "primary", "v6"),
             "SFT-v2 adapter on the base model, test (cross-evaluation, descriptive)": ("SFT-v2 adapter on the base model", "cross-evaluation", "test"),
             "GRPO seed 2, stress": ("GRPO seed 2", "follow-up", "stress"), "GRPO seed 3, V6": ("GRPO seed 3", "follow-up", "v6"),
             "RFT, stress": ("RFT", "follow-up", "stress"), "SFT-v2, stress (follow-up)": ("SFT-v2", "follow-up", "stress"),
             "SFT-v2 + prompt reminder, stress": ("SFT-v2 + prompt reminder", "controls", "stress"),
             "SFT-v2 + prompt reminder, test": ("SFT-v2 + prompt reminder", "controls", "test")}
    listed = set()
    for row in t["S14"][1:]:
        key = cells[row[0]]
        listed.add(key)
        run = d.run(*key)
        ck.cell(f"Table S17 {row[0]} successes", sum(r["full_success"] == "true" for r in run.values()), row[1])
        ck.cell(f"Table S17 {row[0]} false passes", ", ".join(sorted(c for c, r in run.items() if r["known_false_pass"] == "true")), row[2])
    others = [k for k, run in d.runs.items() if k not in listed and any(r["known_false_pass"] == "true" for r in run.values())]
    ck.cell("Table S17 no other cell has a known false pass", len(others), "0")

    # Table 1: greedy full success by model and set (draft Table S20)
    head = t["S20"][0]
    for row in t["S20"][1:]:
        if row[0].startswith("*Exploratory"):
            continue
        for i, h in enumerate(head[1:], 1):
            s = SETS[re.sub(r" \(\d+\)", "", h).lower()]
            if row[i] == "–":
                if (who(row[0], s)[0], who(row[0], s)[1], s) in d.runs:
                    ck.bad.append(f"Table 1 {row[0]} {s}: printed as not evaluated, but the CSV has rows")
                continue
            run = d.run(*who(row[0], s), s)
            got = sum(r["full_success"] == "true" for r in run.values())
            ck.cell(f"Table 1 {row[0]} {s}", f"{got} of {len(run)}" if "of" in row[i] else got, row[i])


def check_text(d: Data, ck: Checker):
    import final_analysis
    import followup_analysis
    from ap_sim.generator.splits import first_test_cases
    S = d.success
    P = lambda s, a, b: metrics.paired_bootstrap(S(*b, s), S(*a, s))    # noqa: E731  a minus b

    def disc(s, a, b):
        x, y = S(*b, s), S(*a, s)
        return sum(y[c] > x[c] for c in x), sum(y[c] < x[c] for c in x)

    sft, sft_fu, sft_ct = ("SFT-v2", "primary"), ("SFT-v2", "follow-up"), ("SFT-v2", "controls")
    g1, g1f, g2, g3 = ("GRPO seed 1", "primary"), ("GRPO seed 1", "follow-up"), ("GRPO seed 2", "follow-up"), ("GRPO seed 3", "follow-up")
    rft, rft_fu = ("RFT", "primary"), ("RFT", "follow-up")
    rem, sec = ("SFT-v2 + prompt reminder", "controls"), ("Second SFT run", "controls")

    # §5.1: the pre-registered comparison and McNemar; S2.5: GRPO seed 1 minus RFT and RFT minus SFT-v2; §5.5 and S2.1: the first 100
    ck.cell("§5.1 GRPO seed 1 − SFT-v2, test", interval(P("test", g1, sft), True), "+7.5 [+4.0, +11.5]")
    gained, lost = disc("test", g1, sft)
    ck.cell("§5.1 gained / lost", f"{gained} / {lost}", "15 / 0")
    ck.cell("§5.1 exact McNemar p", f"{mcnemar_exact(gained, lost):.1e}", "6.1e-05")
    ck.cell("S2.5 GRPO seed 1 − RFT, test", interval(P("test", g1, rft), True), "+3.0 [+1.0, +5.5]")
    ck.cell("S2.5 RFT − SFT-v2, test", interval(P("test", rft, sft), True), "+4.5 [+1.0, +8.5]")
    first = first_test_cases(100)
    ck.cell("§5.5 Claude, first 100 test cases", int(sum(S("Claude Sonnet 5.5", "reference", "test").values())), "99")
    ck.cell("§5.5 Claude cases are the first 100", sorted(S("Claude Sonnet 5.5", "reference", "test")) == sorted(first), "True")
    ck.cell("S2.1 GRPO seed 1, first 100 test cases", int(sum(S(*g1, "test")[c] for c in first)), "100")
    ck.cell("S2.1 first 100: SFT-v2, controls' SFT-v2, RFT, seeds 1–3, second SFT, reminder",
            " ".join(str(int(sum(S(*k, "test")[c] for c in first))) for k in (sft, sft_ct, rft, g1, g2, g3, sec, rem)), "91 94 98 100 99 98 100 98")

    # §5.2: replication (seeds against SFT-v2 on test and the reserve); S2.5: RFT on the reserve
    for name, k in (("seed 1", g1), ("seed 2", g2)):
        ck.cell(f"§5.2 GRPO {name} − SFT-v2 interval above zero, test", P("test", k, sft)["low"] > 0, "True")
    for name, k in (("seed 1", g1f), ("seed 2", g2)):
        ck.cell(f"§5.2 GRPO {name} − SFT-v2 interval above zero, reserve", P("test_reserve", k, sft_fu)["low"] > 0, "True")
    ck.cell("§5.2 GRPO seed 3 − SFT-v2, test", interval(P("test", g3, sft), True), "+4.0 [−0.5, +8.5]")
    ck.cell("§5.2 GRPO seed 3 − SFT-v2, reserve", interval(P("test_reserve", g3, sft_fu), False), "+4 [−3, +12]")
    ck.cell("S2.5 RFT − SFT-v2, reserve", interval(P("test_reserve", rft_fu, sft_fu), False), "+7 [+1, +14]")
    gained, lost = disc("test_reserve", rft_fu, sft_fu)
    ck.cell("S2.5 RFT reserve gained and lost", f"{gained} {lost}", "9 2")
    ck.cell("S2.5 RFT reserve exact McNemar p", f"{mcnemar_exact(gained, lost):.3f}", "0.065")

    # §5.3: the controls, against SFT-v2 re-evaluated in their session and across sessions against GRPO seed 1
    ck.cell("Table S4 SFT-v2 primary vs controls' SFT-v2, test outcomes that differ", sum(a != b for a, b in zip(*(sorted(S(*k, "test").items()) for k in (sft, sft_ct)))), "4")
    ck.cell("§5.3 second SFT − controls' SFT-v2, test", interval(P("test", sec, sft_ct), True), "+5.5 [+2.0, +9.0]")
    ck.cell("§5.3 reminder − controls' SFT-v2, test", interval(P("test", rem, sft_ct), True), "+4.5 [+2.0, +7.5]")
    ck.cell("§5.3 second SFT − GRPO seed 1, test", interval(P("test", sec, g1), True), "−1.0 [−2.5, 0.0]")
    ck.cell("§5.3 reminder − GRPO seed 1, test", interval(P("test", rem, g1), True), "−2.0 [−4.0, −0.5]")
    gained, lost = disc("test", rem, g1)
    ck.cell("S2.5 reminder vs GRPO seed 1 discordant, all one way", f"{gained} {lost}", "0 4")
    ck.cell("§5.3 reminder vs GRPO seed 1 exact McNemar p", f"{mcnemar_exact(gained, lost):.3f}", "0.125")

    # §5.4: seed 3 on V6
    ck.cell("§5.4 GRPO seed 3 − SFT-v2, V6", interval(P("v6", g3, sft), False), "−5 [−10, −1]")
    gained, lost = disc("v6", g3, sft)
    ck.cell("S2.5 seed 3 V6 discordant", f"{gained} {lost}", "0 5")
    ck.cell("§5.4 seed 3 V6 exact McNemar p", f"{mcnemar_exact(gained, lost):.4f}", "0.0625")

    # S2.6: per-seed discordant cases against SFT-v2 (GRPO only / SFT-v2 only)
    ck.cell("S2.6 test discordant, seeds 1–3", " ".join("%d/%d" % disc("test", k, sft) for k in (g1, g2, g3)), "15/0 15/1 15/7")
    ck.cell("S2.6 reserve discordant, seeds 1–3", " ".join("%d/%d" % disc("test_reserve", k, sft_fu) for k in (g1f, g2, g3)), "10/0 10/0 9/5")
    failed = [c for c, v in S(*sft, "test").items() if not v]
    ck.cell("S2.6 every seed succeeded on SFT-v2's 15 test failures", all(S(*k, "test")[c] for k in (g1, g2, g3) for c in failed) and len(failed) == 15, "True")

    # S2.2: the unseen-signature subset (pre-registered secondary), reweighted to the test mix, from the frozen manifests
    table = final_analysis.signature_table(("test",))
    u = final_analysis.unseen_analysis({"sft-v2": d.records(*sft, "test"), "grpo": d.records(*g1, "test")}, table)["unseen"]
    ck.cell("S2.2 unseen-signature cases", u["cases"], "48")
    diff = u["difference"]
    ck.cell("S2.2 unseen GRPO − SFT-v2, reweighted", f"{100 * diff['rate']:+.1f} [{100 * diff['low']:+.1f}, {100 * diff['high']:+.1f}]", "+15.0 [+8.3, +21.7]")
    ck.cell("S2.2 unseen successes, GRPO and SFT-v2", " ".join(str(int(sum(S(*k, "test")[c] for c in table if table[c]["unseen"]))) for k in (g1, sft)), "48 41")
    v5 = [c for c in table if table[c]["unseen"] and table[c]["type"] == "V5"]
    share = final_analysis.TEST_MIX["V5"] * sum(S(*g1, "test")[c] - S(*sft, "test")[c] for c in v5) / len(v5)
    ck.cell("S2.2 unseen V5 cases and their points", f"{len(v5)} {100 * share:.1f}", "9 13.3")
    for k, want8 in ((sft, "1.00"), (rft, "1.00"), (g1, "1.00")):
        n = [int(r["sampled_successes_of_8"]) for r in d.run(*k, "test").values()]
        ck.cell(f"S2.2 pass@8 {k[0]}", f"{sum(x > 0 for x in n) / len(n):.2f}", want8)

    # S2.5: the 21 stress pairs, and the 57 per-group stress intervals of the follow-up's breakdowns
    import itertools
    seven = [sft_fu, rft_fu, g1f, g2, g3, rem, sec]
    pairs = [(a, b) for a, b in itertools.combinations(seven, 2)]
    excl = []
    for a, b in pairs:
        r = P("stress", b, a)
        if r["low"] > 0 or r["high"] < 0:
            excl.append((a, b, r))
    ck.cell("S2.5 stress pairs", len(pairs), "21")
    ck.cell("S2.5 stress pairs excluding zero", len(excl), "1")
    if len(excl) == 1:
        a, b, r = excl[0]
        names = {a[0], b[0]}
        ck.cell("S2.5 that pair", sorted(names), str(sorted({"Second SFT run", "GRPO seed 2"})))
        x = P("stress", sec, g2)
        ck.cell("S2.5 second SFT − GRPO seed 2, stress", interval(x, False), "−9 [−17, −1]")
        ck.cell("S2.5 its exact McNemar p", f"{mcnemar_exact(*disc('stress', sec, g2)):.3f}", "0.049")
    models = {"sft-v2": d.records(*sft_fu, "stress"), "rft": d.records(*rft_fu, "stress"), "grpo-20261101": d.records(*g1f, "stress"),
              "grpo-20261107": d.records(*g2, "stress"), "grpo-20261108": d.records(*g3, "stress")}
    rep = followup_analysis.stress_report(models)
    ivs = list(rep["difference_vs_sft_v2"].values())
    groups = {"difference_vs_sft_v2": len(ivs)}
    for kind in followup_analysis.DIFFERENCES_BY:
        block = [x for g in rep[f"difference_by_{kind}"].values() for x in g.values()]
        groups[kind] = len(block)
        ivs += block
    ck.cell("S2.5 per-group stress intervals (vs SFT-v2, format, complication x format, combination)",
            " ".join(str(groups[k]) for k in ("difference_vs_sft_v2", "format", "complication_x_format", "combination")), "3 6 18 30")
    ex = [x for x in ivs if x["low"] > 0 or x["high"] < 0]
    ck.cell("S2.5 per-group stress intervals excluding zero", len(ex), "1")
    combos = [(g, m) for g, by in rep["difference_by_combination"].items() for m, x in by.items() if x["low"] > 0 or x["high"] < 0]
    if len(combos) == 1:
        g, m = combos[0]
        ids = followup_analysis.stress_groups()["combination"][g]
        a = {c: v for c, v in S(*sft_fu, "stress").items() if c in ids}
        b = {c: v for c, v in S(*g1f, "stress").items() if c in ids}
        gained, lost = sum(b[c] > a[c] for c in a), sum(b[c] < a[c] for c in a)
        ck.cell("S2.5 that group: model, cases, gained to lost, exact McNemar p", f"{m} {len(ids)} {gained} to {lost} {mcnemar_exact(gained, lost):.3f}", "grpo-20261101 12 4 to 0 0.125")

    # Figure 2: configuration minus SFT-v2, percentage points with 95% paired-bootstrap intervals
    for s, label, k, base, want in FIGURE_1:
        ck.cell(f"Figure 2 {s} {label}", interval(P(s, tuple(k), tuple(base)), s == "test"), want)

    # S2.7: run and episode counts
    runs = {k for k in d.runs if k[1] not in ("reference", "post hoc")}
    ck.cell("S2.7 stored greedy runs and episodes", f"{len(runs)} {sum(len(d.runs[k]) for k in runs)}", "39 4520")
    ck.cell("S2.7 test episodes with an attempted unsafe action, any model", sorted({k[0] for k in runs if k[2] == "test" and any(int(r["attempted_unsafe"]) for r in d.runs[k].values())}),
            str(sorted({"Base", "RFT"})))


FIGURE_1 = [  # the final paper's Figure 2 (draft Figure 1): (set, row, configuration, SFT-v2 baseline, printed interval)
    ("test", "GRPO seed 1 (pre-registered)", ("GRPO seed 1", "primary"), ("SFT-v2", "primary"), "+7.5 [+4.0, +11.5]"),
    ("test", "GRPO seed 2", ("GRPO seed 2", "follow-up"), ("SFT-v2", "primary"), "+7.0 [+3.5, +11.0]"),
    ("test", "GRPO seed 3", ("GRPO seed 3", "follow-up"), ("SFT-v2", "primary"), "+4.0 [−0.5, +8.5]"),
    ("test", "RFT", ("RFT", "primary"), ("SFT-v2", "primary"), "+4.5 [+1.0, +8.5]"),
    ("test", "Second SFT run", ("Second SFT run", "controls"), ("SFT-v2", "controls"), "+5.5 [+2.0, +9.0]"),
    ("test", "SFT-v2 + prompt reminder", ("SFT-v2 + prompt reminder", "controls"), ("SFT-v2", "controls"), "+4.5 [+2.0, +7.5]"),
    ("test_reserve", "GRPO seed 1", ("GRPO seed 1", "follow-up"), ("SFT-v2", "follow-up"), "+10 [+4, +16]"),
    ("test_reserve", "GRPO seed 2", ("GRPO seed 2", "follow-up"), ("SFT-v2", "follow-up"), "+10 [+4, +16]"),
    ("test_reserve", "GRPO seed 3", ("GRPO seed 3", "follow-up"), ("SFT-v2", "follow-up"), "+4 [−3, +12]"),
    ("test_reserve", "RFT", ("RFT", "follow-up"), ("SFT-v2", "follow-up"), "+7 [+1, +14]"),
    ("stress", "GRPO seed 1", ("GRPO seed 1", "follow-up"), ("SFT-v2", "follow-up"), "+4 [−5, +14]"),
    ("stress", "GRPO seed 2", ("GRPO seed 2", "follow-up"), ("SFT-v2", "follow-up"), "+8 [−1, +17]"),
    ("stress", "GRPO seed 3", ("GRPO seed 3", "follow-up"), ("SFT-v2", "follow-up"), "+4 [−5, +13]"),
    ("stress", "RFT", ("RFT", "follow-up"), ("SFT-v2", "follow-up"), "+6 [−2, +14]"),
]

PAPER_TABLES = r"""
**Table 1.** Right final action and complete-task success come from the evaluations. The false-pass and heuristic columns are post-hoc analyses, not confirmatory measures.

| Set and model | Right final action | Complete-task success | Known false passes | Heuristic, right final action |
|---|---|---|---|---|
| *Test (200 cases)* | | | | 200 |
| SFT-v2 | 200 | 185 | 0 | |
| RFT | 198 | 194 | 0 | |
| GRPO seed 1 | 200 | 200 | 0 | |
| GRPO seed 2 | 200 | 199 | 0 | |
| GRPO seed 3 | 200 | 193 | 0 | |
| *Exploratory, post hoc:* | | | | |
| Second SFT run | 200 | 198 | 0 | |
| SFT-v2 + prompt reminder | 200 | 196 | 2 | |
| SFT-v2, re-evaluated | 200 | 187 | 0 | |
| *V6 (100 cases)* | | | | 100 |
| SFT-v2 | 99 | 43 | 1 | |
| GRPO seed 1 | 100 | 44 | 0 | |
| GRPO seed 2 | 99 | 44 | 0 | |
| GRPO seed 3 | 99 | 38 | 1 | |
| *Policy shift (44 cases)* | | | | 44 |
| SFT-v2 | 44 | 23 | 0 | |
| GRPO seed 1 | 43 | 22 | 0 | |
| GRPO seed 2 | 43 | 22 | 0 | |
| GRPO seed 3 | 44 | 23 | 0 | |
| *Stress (100 cases)* | | | | 100 |
| SFT-v2 | 99 | 80 | 1 | |
| RFT | 98 | 86 | 1 | |
| GRPO seed 1 | 96 | 84 | 0 | |
| GRPO seed 2 | 97 | 88 | 1 | |
| GRPO seed 3 | 99 | 84 | 0 | |
| *Exploratory, post hoc:* | | | | |
| Second SFT run | 98 | 79 | 0 | |
| SFT-v2 + prompt reminder | 96 | 83 | 1 | |

**Table S6.** Success, right final action and report validity, every greedy evaluation run.

| Model | Set | Success | Right final action | Report valid |
|---|---|---|---|---|
| Base | test / V6 / policy shift / forgetting | 0 / 0 / 0 / 0 | 45 / 6 / 32 / 29 (outcome correct: 40 / 3 / 27 / 25) | 0 / 0 / 0 / 0 |
| SFT-v2 | test / reserve / stress | 185 / 89 / 80 | 200 / 100 / 99 | 185 / 89 / 80 |
| SFT-v2 | V6 / policy shift / forgetting | 43 / 23 / 40 | 99 / 44 / 40 | 43 / 23 / 40 |
| RFT | test / reserve / stress | 194 / 96 / 86 | 198 / 99 / 98 | 194 / 96 / 86 |
| GRPO seed 1 | test / reserve / stress | 200 / 99 / 84 | 200 / 100 / 96 | 200 / 99 / 84 |
| GRPO seed 1 | V6 / policy shift / forgetting | 44 / 22 / 40 | 100 / 43 / 40 | 44 / 22 / 40 |
| GRPO seed 2 | test / reserve / stress | 199 / 99 / 88 | 200 / 100 / 97 | 199 / 99 / 88 |
| GRPO seed 2 | V6 / policy shift / forgetting | 44 / 22 / 40 | 99 / 43 / 40 | 44 / 22 / 40 |
| GRPO seed 3 | test / reserve / stress | 193 / 93 / 84 | 200 / 100 / 99 | 193 / 93 / 84 |
| GRPO seed 3 | V6 / policy shift / forgetting | 38 / 23 / 32 | 99 / 44 / 40 | 38 / 23 / 32 |
| Second SFT run | test / stress | 198 / 79 | 200 / 98 | 198 / 79 |
| SFT-v2 + prompt reminder | test / stress | 196 / 83 | 200 / 96 | 196 / 83 |
| SFT-v2, re-evaluated in the controls' session | test | 187 | 200 | 187 |

**Table S7.** Primary evaluation, test success by family, sampled success and pass^8.

| Primary evaluation, test | V1 (40) | V2 (36) | V3a (20) | V3b (20) | V4 (44) | V5 (40) | Sampled success | pass^8 | Sampled success on V4 |
|---|---|---|---|---|---|---|---|---|---|
| Base | 0 | 0 | 0 | 0 | 0 | 0 | – | – | – |
| SFT-v2 | 39 | 36 | 19 | 20 | 44 | 27 | 0.889 | 0.63 | 0.832 |
| RFT | 40 | 36 | 19 | 20 | 43 | 36 | 0.904 | 0.675 | 0.747 |
| GRPO seed 1 | 40 | 36 | 20 | 20 | 44 | 40 | 0.980 | 0.87 | 0.969 |

**Table S8.** Stress failures by primary category.

| Model | Failures | Wrong decision | Canceled-invoice omission | Derived-figure failure | Other |
|---|---|---|---|---|---|
| SFT-v2 | 20 | 1 | 13 | 6 | 0 |
| Second SFT run | 21 | 2 | 5 | 14 | 0 |
| SFT-v2 + prompt reminder | 17 | 4 | 5 | 7 | 1 |
| RFT | 14 | 2 | 3 | 9 | 0 |
| GRPO seed 1 | 16 | 4 | 0 | 10 | 2 |
| GRPO seed 2 | 12 | 3 | 0 | 8 | 1 |
| GRPO seed 3 | 16 | 1 | 1 | 10 | 4 |

**Table S9.** Every label on the stress failures.

| Model | Failures | Wrong decision | Missing evidence (all canceled-invoice) | Derived figure | Other citation problem | Discrepancy type wrong | No flag |
|---|---|---|---|---|---|---|---|
| SFT-v2 | 20 | 1 | 13 | 7 | 0 | 2 | 0 |
| Second SFT run | 21 | 2 | 5 | 14 | 0 | 2 | 0 |
| SFT-v2 + prompt reminder | 17 | 4 | 5 | 7 | 2 | 3 | 0 |
| RFT | 14 | 2 | 3 | 9 | 0 | 2 | 0 |
| GRPO seed 1 | 16 | 4 | 0 | 10 | 1 | 3 | 1 |
| GRPO seed 2 | 12 | 3 | 0 | 8 | 0 | 3 | 1 |
| GRPO seed 3 | 16 | 1 | 1 | 10 | 2 | 3 | 1 |

**Table S10.** Grader sensitivity, by model.

| Set, model | Grader (g8) success | Variant success | (a) / (b) / (o) |
|---|---|---|---|
| V6, SFT-v2 | 43 | 93 | 53 / 2 / 1 |
| V6, GRPO seed 1 | 44 | 97 | 54 / 1 / 1 |
| V6, GRPO seed 2 | 44 | 96 | 56 / 0 / 0 |
| V6, GRPO seed 3 | 38 | 92 | 54 / 3 / 3 |
| Stress, SFT-v2 | 80 | 85 | 5 / 0 / 2 |
| Stress, RFT | 86 | 89 | 3 / 2 / 4 |
| Stress, GRPO seed 1 | 84 | 90 | 6 / 1 / 3 |
| Stress, GRPO seed 2 | 88 | 93 | 5 / 0 / 3 |
| Stress, GRPO seed 3 | 84 | 88 | 4 / 3 / 3 |
| Stress, SFT-v2 + prompt reminder | 83 | 88 | 5 / 1 / 1 |
| Stress, second SFT run | 79 | 84 | 5 / 4 / 5 |

**Table S11.** Other paired comparisons.

| Comparison | Test | Reserve | Stress |
|---|---|---|---|
| GRPO seed 1 − RFT | +3.0 [+1.0, +5.5] | +3 [−1, +7] | −2 [−10, +6] |
| GRPO seed 2 − RFT | +2.5 [0.0, +5.0] | +3 [−1, +7] | +2 [−5, +9] |
| GRPO seed 3 − RFT | −0.5 [−3.5, +2.5] | −3 [−8, +2] | −2 [−8, +4] |
| Second SFT run − SFT-v2 | +6.5 [+3.0, +10.5] | – | −1 [−10, +8] |
| SFT-v2 + prompt reminder − SFT-v2 | +5.5 [+2.5, +9.0] | – | +3 [−6, +12] |
| Second SFT run − (SFT-v2 + prompt reminder) | +1.0 [−1.5, +3.5] | – | −4 [−13, +5] |

**Table S12.** Decision errors and attempted unsafe actions, greedy runs.

| Model | Test: incorrect hold / no counted final action | Stress: incorrect hold / no counted final action | Test episodes with an attempted unsafe action (attempts, executed) | Weighted error per 1,000, test |
|---|---|---|---|---|
| Base | 3 / 152 | – | 7 (7, 0) | 976.56 |
| SFT-v2 | 0 / 0 | 0 / 1 | 0 | 234.48 |
| SFT-v2 + prompt reminder (no training) | 0 / 0 | 2 / 2 | 0 | 234.48 |
| Second SFT run | 0 / 0 | 1 / 1 | 0 | 234.48 |
| RFT | 0 / 2 | 0 / 2 | 1 (9, 0) | 249.01 |
| GRPO seed 1 | 0 / 0 | 2 / 2 | 0 | 234.48 |
| GRPO seed 2 | 0 / 0 | 2 / 1 | 0 | 234.48 |
| GRPO seed 3 | 0 / 0 | 0 / 1 | 0 | 234.48 |

**Table S14.** The false passes the audit detected.

| Cell | Successes | False passes |
|---|---|---|
| SFT-v2, V6 (primary evaluation) | 43 | v6-V4-10 |
| SFT-v2 adapter on the base model, test (cross-evaluation, descriptive) | 183 | test-V4-4 |
| GRPO seed 2, stress | 88 | stress-V4-13 |
| GRPO seed 3, V6 | 38 | v6-V4-10 |
| RFT, stress | 86 | stress-V4-18 |
| SFT-v2, stress (follow-up) | 80 | stress-V4-18 |
| SFT-v2 + prompt reminder, stress | 83 | stress-V4-18 |
| SFT-v2 + prompt reminder, test | 196 | test-V1-26, test-V4-4 |

**Table S20.** Greedy complete-task success, by model and evaluation set. A dash means not evaluated. The reference row's test column counts the first 100 test cases. Claude Sonnet 5.5 ran at its API's default sampling, which cannot be set, so its row is not greedy.

| Model | Test (200) | Reserve (100) | Stress (100) | V6 (100) | Policy shift (44) | Forgetting (40) |
|---|---|---|---|---|---|---|
| Base (Qwen3-4B) | 0 | – | – | 0 | 0 | 0 |
| SFT-v2 | 185 | 89 | 80 | 43 | 23 | 40 |
| RFT | 194 | 96 | 86 | – | – | – |
| GRPO seed 1 (pre-registered) | 200 | 99 | 84 | 44 | 22 | 40 |
| GRPO seed 2 | 199 | 99 | 88 | 44 | 22 | 40 |
| GRPO seed 3 | 193 | 93 | 84 | 38 | 23 | 32 |
| *Exploratory, post hoc:* | | | | | | |
| Second SFT run | 198 | – | 79 | – | – | – |
| SFT-v2 + prompt reminder (no training) | 196 | – | 83 | – | – | – |
| SFT-v2, re-evaluated in the controls' session | 187 | – | – | – | – | – |
| *Reference:* Claude Sonnet 5.5 | 99 of 100 | – | 100 | – | – | – |
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(ROOT / "data" / "outcomes.csv"))
    ap.add_argument("--show", action="store_true", help="print every check, not only failures")
    ap.add_argument("--rerun-heuristic", action="store_true",
                    help="also rerun the scripted heuristic (scripts/posthoc/review_item4.py's agent) on test, V6, the policy shift and stress, "
                         "through the environment and the grader, and check every case against the CSV")
    a = ap.parse_args()
    d = Data(pathlib.Path(a.csv))
    ck = Checker()
    check_tables(d, ck, a.show)
    check_text(d, ck)
    if a.rerun_heuristic:
        sys.path.insert(0, str(ROOT / "scripts/posthoc"))
        import posthoc_common
        import review_item4      # its main() also checks an anchor on stored transcripts, which are not released; its agent needs none
        for s in ("test", "v6", "policy_shift", "stress"):
            factories = posthoc_common.cases(s)
            rows = {case: review_item4.episode(case, f) for case, f in factories.items()}
            got = {c: "true" if r["right"] else "false" for c, r in rows.items()}
            want = {c: r["right_final_action"] for c, r in d.run("Scripted heuristic", "post hoc", s).items()}
            ck.cell(f"heuristic rerun {s}: right final action, every case", got == want, "True")
            ck.cell(f"heuristic rerun {s}: attempted and executed unsafe actions", sum(r["attempted_unsafe"] + r["executed_unsafe"] for r in rows.values()), "0")
    print(f"checked {ck.ok + len(ck.bad)} values: {ck.ok} equal, {len(ck.bad)} differ")
    for why, n in sorted(ck.skipped.items()):
        if n:
            print(f"  skipped {n}: {why}")
    for b in ck.bad:
        print("  DIFFERS " + b)
    print("Not recomputable from this release: Table S16 (costs), Table S5 (the learning-rate pilot), Table S22 (Inkling 2 Small) and the paper's other "
          "aggregate-only figures; see README.md.")
    raise SystemExit(1 if ck.bad else 0)


if __name__ == "__main__":
    main()

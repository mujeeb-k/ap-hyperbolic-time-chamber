# Messiness stress set: parameters and timing

The 100 cases in `messiness.json` are the follow-up study's stress set, specified on 2026-11-06, before any stress result was read. Each entry stores only parameters, and each case is rebuilt from them.

This file says how to turn a case's `params` into the records the agent sees and the hidden world. Read it together with `docs/POLICY.md`. Every rule of POLICY applies unchanged; the only new thing here is the timed parameter format. The generator code is `src/ap_sim/stress/messiness.py`, but this file is meant to suffice without it.

## 1. Two parameter formats

Each case is in one of two formats. You can tell them apart from `params` alone: a timed case has the field `timed_receipts`.

- **Frozen format.** This is exactly the parameter set of the existing splits (`receipts`, `other_invoices`, `cancelled_invoices`, `cancelled_at`, `reversals`, …).
  - It is built by the frozen `fixtures.build` and has the same timing as every test case: receipt i at −2400 + 400 i, other invoices every 120 minutes after the last receipt with the cancelled one first, and one shared `cancelled_at`.
  - It is used whenever that timing can express the case: at most 3 receipts, at most one cancelled invoice, and no M3. That covers M1 with 2 or 3 receipts, M4, and their combination, as data.
  - `reversals` is always empty and the policy is `sandbox-v1`.
- **Timed format.** It is used only for what `fixtures.build` cannot express:
  - several cancellations at different times (M2);
  - 4 or 5 receipts (M1);
  - a re-issue posted after its original's cancellation (M3). In the frozen timing the shared cancellation comes after every posting, so that order cannot be written.

  Every pre-episode time is stored. §2 to §4 describe this format.

## 2. Timed parameters

Each timed entry's `params` has exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `po_qty` | int | Ordered quantity of the PO's only line (EA). The PO is open. |
| `price` | decimal string | Unit price in EUR. Applies to the PO line and to every invoice line. |
| `invoice_qty` | int | Quantity on the invoice under review. It has one line, on the same PO line. |
| `blocked_at` | int | When the invoice under review was posted and evaluated (it was found blocked then). |
| `timed_receipts` | list of `[quantity, posted_at]` | The recorded goods receipts on the PO line, in time order. Every one is a receipt movement. There are no reversals. |
| `timed_other_invoices` | list of `[quantity, posted_at, cancelled_at]` | The other invoices on the PO line, in order of `posted_at`. If `cancelled_at` is `null`, the invoice is active (`posted_released`). Otherwise it is `cancelled`. |
| `physical_total` | int | Hidden: the quantity physically delivered by t = 0. |
| `delay` | int or `null` | Hidden: how long after a delivery-confirmation request the warehouse answers (business minutes). `null` means it never answers. |

- Times are business minutes from the episode start. t = 0 is D1 09:00, a business day is 480 minutes, and negative values are history before the episode (POLICY C1).
- The policy profile is always `sandbox-v1`.
- There is no `policy` field and there are no other fields.

**Building the records** (the same as `fixtures.build`, except that the times are read, not computed):

- **PO:** one PO, supplier "Supplier A", status open. It has one line with `po_qty` at `price`, and line amount = `po_qty` × `price`, rounded to the cent.
- **Receipts:** one record per `timed_receipts` element, with that quantity and `posted_at`.
- **Other invoices:** one invoice per `timed_other_invoices` element, with one line of that quantity at `price` (amount = quantity × price, rounded to the cent).
  - Each has `posted_at`, eligibility `eligible`, and `evaluated_at` = `posted_at`.
  - An active one has state `posted_released`, version 1.
  - A cancelled one has state `cancelled` with its own `cancelled_at`, version 2 (posted, then cancelled).
- **The invoice under review:** state `posted_blocked`, `posted_at` = `evaluated_at` = `blocked_at`, one line of `invoice_qty` at `price`.
  - The comparison results shown at the start are those of its evaluation at `blocked_at`, computed on the records as they stood then.
  - Records as they stood then: receipts posted at or before `blocked_at`, and invoices posted at or before it. An invoice cancelled later is shown as it was then: active, version 1.
- **Hidden world** (both formats):
  - Each recorded receipt is a physical delivery of its quantity at its `posted_at`.
  - If `physical_total` exceeds the sum of the receipts, the difference was delivered at t = −300 and not recorded.
  - A confirmation request made at time T is answered at T + `delay`, with the delivered total at that time. Every delivery is before t = 0, so that total is `physical_total`.
- **Record ids** come from the case's seed and are irrelevant to the outcome.

## 3. Timed-format timing rules (all hold for every timed case)

1. **Receipts:** 1 to 5. Receipt i (1-based) is posted at −2400 + 400 i, which gives −2000, −1600, −1200, −800 and −400. A sixth would post at t = 0 and is never used.
2. **The block:**
   - `blocked_at` is −160;
   - or, in a stale-receipt case, 100 minutes before the last receipt (`timed_receipts[-1][1] − 100`). The last receipt was then posted after the block's evaluation.
3. **Other invoices:**
   - Every other invoice is posted after the first receipt and before `blocked_at`. No invoice is posted after the block.
   - At its posting, the receipts already posted covered it together with every invoice then active (as a released invoice would have been). This is a realism rule of the generator; the grader does not use it.
4. **Cancellations:** each `cancelled_at` is later than its own `posted_at`. It is either before `blocked_at`, or after it (a later cancellation, which makes the block stale).
5. **Before the episode:** every event (receipt, posting, cancellation, block) is at or before t = −20, and at least 20 minutes from any other event. No two events share a time. The generator has enforced this since the set was rebuilt on 2026-11-06 (§4, M3), and the loader refuses a case that breaks it; the frozen-format cases meet it too.
6. **Hidden delivery:** the unrecorded delivery, if any, is at t = −300.

## 4. Complications, by content (both formats)

Each case carries one or two of the four, and the entry's `complications` lists them. They are decided from `params` alone. A frozen case is read with its frozen times (§1).

- **M1, several receipts:** 2 or more receipts (2 to 5). Each is a partial delivery: together they are less than `po_qty`.
  - In a shortfall case they also cover less than the invoice needs.
  - A partly confirmed delivery (unrecorded quantity between 0 and the gap) never occurs.
- **M2, several cancellations:** at least two cancelled other invoices (2 or 3), each with a different `cancelled_at`. Always timed.
- **M3, a re-issued invoice:** an active other invoice with the same quantity as a cancelled other invoice, posted **after that invoice's cancellation**. On 2026-11-06, before any model was evaluated on the set, the set was rebuilt so that every M3 case meets this.
  - The original is cancelled before the block, and the re-issue is posted between that cancellation and the block.
  - No case has an active invoice with a cancelled invoice's quantity that was posted while that invoice was still active.
  - M3 cases are always in the timed format.
- **M4, a look-alike duplicate:** an active other invoice with the same quantity as `invoice_qty`.

Without a complication, the case has none of its content:
- one receipt;
- at most one cancelled invoice;
- no active invoice with the quantity of an earlier-posted cancelled one;
- no active invoice with `invoice_qty`.

Ordinary other active invoices (0 to 2, of other quantities) and one ordinary cancellation can appear in any case. No rule treats a re-issue or a look-alike differently: every active other invoice consumes coverage (POLICY A4), and every cancelled one consumes none.

## 5. Type and outcome (derived, never chosen)

All quantities below are at t = 0.
- coverage = net receipts − active other invoices;
- gap = `invoice_qty` − coverage, floored at 0;
- unrecorded = `physical_total` − net receipts.

The type follows from these (frozen `causal.derive_variant`):

| Type | gap | unrecorded | `delay` | Outcome (g7) | discrepancy_type |
|---|---|---|---|---|---|
| V5 | 0 | 0 | any | release | none |
| V1 | > 0 | = gap or > gap | 240 or 720 | release | receipt_coverage_gap |
| V2 | > 0 | 0 | 240 or 720 | keep_blocked | receipt_coverage_gap |
| V3a | > 0 | = gap or > gap | 1920 | release | receipt_coverage_gap |
| V3b | > 0 | 0 | 1920 | keep_blocked | receipt_coverage_gap |
| V4 | > 0 | 0, = gap or > gap | 2880 or `null` | pending | receipt_coverage_gap |

- **V5:** the block is stale. At `blocked_at` coverage was short (a receipt was posted later, or an invoice was cancelled later), and at t = 0 it covers the invoice.
- **Stored labels:** each entry's `variant`, `expected` and `discrepancy_type` were computed this way on the built records, and the outcome by the frozen grader g7 on an unacted surfaces/4 log.

## 6. Scope (checked on every case)

What the checks prove: the build's scope check recomputes the labels with the same code that made them, so it shows determinism, not correctness. The correctness evidence is an independent verifier and two independent recomputations of the labels, made with separate code during development. That code is not part of this release. The page test (`test_pages_at_reset_mirror_the_parameters` in `tests/test_stress_set.py`) checks every page at reset against the parameters in both formats, including the receipts with their quantities, the PO line, the invoice line and the blocking comparisons, recomputed from the parameters without `derive()`.

The frozen `causal.check_scope` covers frozen-format cases. Its three assertions are reimplemented on the timed builder's records for timed cases.

- **At t = 0:**
  - net receipts are positive;
  - the only blocking comparison that fails is invoice vs receipt (none fails in V5);
  - there is no partial confirmation.
- **At `blocked_at`:** on the records as they stood then, invoice vs receipt fails and no other blocking comparison fails.
- **Never present:**
  - reversals, credit notes, unit conversions or price differences;
  - a second PO line, or a receipt posted during the episode;
  - a confirmation request before the episode.

## 7. Seeds, strata and design

- **Seeds:**
  - The seed is 11,800,000 + 10,000 × group + k, with groups 0 to 5 = V1, V2, V3a, V3b, V4, V5.
  - Each stratum draws k = 0, 1, … until it holds its count: V1 20, V2 18, V3a 10, V3b 10, V4 22, V5 20.
- **The design:**
  - Each single complication is in 7 cases and each pair in 12, so each complication is in 43.
  - The design is spread over the types in proportion, as a controlled rounding (`allocation`).
  - Four slot swaps follow (`DESIGN_SWAPS`), because V5 cannot hold M3 or M3+M4. In V5 the re-issued original is cancelled before the block, so without M1 (a later receipt) or M2 (another, later cancellation) nothing is left to make the block stale.
    - V5's two M3 slots swap with a V4 and a V2 M1+M3 slot.
    - V5's two M3+M4 slots swap with a V4 and a V1 M2+M3 slot.
    - Every cell total, and so every complication and pair count, is unchanged.
- **Order:** `build_splits.interleave` with seed 20261106.
- **Draws that were not kept:** they are listed in `build_summary.json`.
  - A draw is redrawn on a derived-type mismatch or a key collision with any split.
  - A draw is withdrawn when the reference agent does not solve it.

## 8. Known properties

- **M1's receipt counts lean toward 5.** The 43 M1 cases have 2, 3, 4 and 5 receipts in 5, 9, 5 and 24 cases, although the count is drawn uniformly: rejection sampling keeps more 5-receipt draws. An earlier build of the set had the same skew (8, 10, 4, 21). Five-receipt cases release in 16 of 24, against 24 of 43 across M1 (p ≈ 0.10, not significant).
- **The loader's leakage refusal is exact-match only.** It refuses a shared seed and any exact match on the parameter, visible or figure keys, but a near variant of an existing case (PO quantity +1, price +€0.01, a receipt split in two) would load. Near variants were covered at build time by a scan for cases with the same core (`same_core`, within 5%), which found none. The scan's script is not part of this release.

## 9. Limitations (recorded 2026-11-07, no case changed)

1. **The grader does not check the quantity posted (POLICY G-11).**
   - **The rule:** a receipt posting must rest on an answered delivery confirmation, and the grader checks only that (`release_without_delivery_basis` in `src/ap_sim/grader/grader.py`). It does not check the quantity.
   - **The engine** refuses only a request above "confirmed delivered total − net posted" (the posting-request check in `src/ap_sim/engine/engine.py`).
   - **So posting only the shortfall passes**, and leaves delivered units unrecorded. Shown on stress-V1-14: the gap is 24 and 29 units are unrecorded. Posting 29 and posting 24 both score 1 under g7 and g8, with no critical failure, on 2026-11-07.
   - **The rule the agent follows** (G-11) asks for all confirmed, unrecorded units. The score cannot tell an agent that follows it from one that posts just enough.
2. **The look-alike invoices (M4) test careful reading and citation, not duplicate-invoice detection.**
   - The policy has no duplicate-payment rule. A look-alike is another active invoice of the same quantity on the same PO line, and it consumes coverage like any other invoice.
   - A real AP clerk would check an identical invoice on the same PO line. For example, stress-V3b-6 has a 64-unit blocked invoice beside another active 64-unit invoice.
   - The rules the model sees require release when coverage allows it.
3. **The structure is synthetic.**
   - **Receipt times:** receipts sit on a fixed time grid across cases. Receipt i always posts at −2400 + 400·i minutes, and in the stress set every receipt is at one of −2000, −1600, −1200, −800 or −400.
   - **One supplier and one item:** each case has one supplier ("Supplier A") and one PO line ("Item 1").
   - **Prices:** unit prices repeat. The stress set's 100 cases use 9 distinct unit prices.
4. **A re-issued invoice carries no reference to the invoice it replaces.**
   - The invoice records have no "replaces" or reference field: their fields are id, po, supplier, state, posted_at, cancelled_at, version and the eligibility fields.
   - A re-issue is recognisable only by its quantity and its timing: posted after a cancelled invoice of the same quantity, and after that one's cancellation.

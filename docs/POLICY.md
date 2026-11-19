# POLICY

This is the simulated accounts-payable policy that the environment enforces. It is a synthetic sandbox policy. It
does not reproduce the configuration of any real organization. All amounts are in euros.

## How to read this file

- Section A states the comparison rules. The engine implements them in `src/ap_sim/domain/policy.py`.
- Section B resolves the ambiguities in those rules, each with a worked example.
- Section C fixes the virtual clock, the counting convention for tool calls and the workflow templates that give
  T_res, the earliest time at which a case can be resolved.
- Sections D and E give the reference variants and the golden examples, with every computed value.
- Section F gives the expected reward of each agent in the agent zoo.
- Section G lists each open question and its status. A G-n row records a question and the policy's answer. An O-n
  row records a further item with its status (resolved, open or noted).
- Section H lists the checks recorded when the policy was fixed, and the checks that were missing then.
- Section I adds the delivered-total rule, `sandbox-v2`'s effective time, the out-of-scope situations, the corrected
  call counts and the report's `discrepancy_type`.

The T_res workflows below are hand-built schedules, not the output of a search (section H).

Version 1 (v1) and version 2 (v2) are versions of the environment and its case sets. Version 1's frozen cases are in
`manifests/dev_cases.json`, `manifests/frozen/` and `manifests/frozen_policy_shift/`. Version 2's are in
`manifests/v2/`. They are unrelated to the policy profiles `sandbox-v1` and `sandbox-v2`.

Grader versions. Some values depend on the grader version. Graders g3–g7 require an accepted `finish` for a
reward. Grader g4 adds the completion contract (G-12). Graders g5–g7 use the corrected workflow templates (O-9,
section I4). Where a T_res value differs, the table gives the value for g5–g7, and the g4 value follows in
brackets ("[g4: 505]"). Grader g6 adds the unsafe release without current eligibility (section H). Grader g7 adds the
report's `discrepancy_type` (I5). The computed values in this document (T_res, call counts, the agent-zoo matrix,
expected outcomes) are given for graders up to g7. Grader g8 (`src/ap_sim/grading/g8.py`), which scored the study's
results, grades as g7 does and adds checks these tables do not reflect. One of them is that a pending disposition
taken before a final check of the delivery confirmations scores 0.

Sections: A. policy rules · B. resolution of ambiguities · C. clock and counting convention · D. variants ·
E. golden examples · F. agent-zoo matrix · G. resolved questions and further items · H. checks recorded when the
policy was fixed · I. additions (2026-10-30).

---

## A. Policy rules (six comparisons A1–A6, the zero-receipt rule A4-DW and the closed-PO rule A7, every field)

Policy version: `sandbox-v1` (the default everywhere). `sandbox-v2` applies only in the policy-shift set (A2,
section I2).

### Boundary note

Every limit below fails only when strictly exceeded, so equality passes. A limit phrased as "exceeding", "by more
than" or "must not exceed" passes at the limit itself. Several cases rely on this at A4 (V1, V3a, V6a, G1, G4, G8
after resolution).

### A1. Price tolerance (10% or €1,000)

| Field | Value |
| --- | --- |
| Applicability condition | Every invoice line on the PO line |
| Formula and units | Unit-price difference as % of PO unit price. Line-level monetary effect = unit difference × invoiced quantity |
| Reference basis | PO unit price |
| Percentage denominator | PO unit price |
| Absolute limit | €1,000 against the line-level monetary effect |
| Both limits must pass, or either suffices | Exceeding either fails (the stricter governs) |
| Boundary behavior (inclusive / exclusive) | Equality passes |
| Effect (blocking / informational / disabled) | Blocking |
| Worked example (inputs → result) | V1, invoice €20.00 vs PO €20.00. Difference 0% and €0 → pass |

### A2. Quantity tolerance (5% or €500)

| Field | Value |
| --- | --- |
| Applicability condition | `sandbox-v2` only, on the invoice-vs-receipt comparison when net receipts > 0 |
| Formula and units | Excess = joint invoiced quantity on the line − available coverage (A4) |
| Reference basis | Available coverage |
| Percentage denominator | Available coverage |
| Absolute limit | €500 against the money conversion = excess quantity × PO unit price |
| Both limits must pass, or either suffices | Exceeding either fails. It passes when excess ≤ 5% of coverage AND excess × PO unit price ≤ €500 |
| Boundary behavior (inclusive / exclusive) | Equality passes |
| Effect (blocking / informational / disabled) | Disabled in `sandbox-v1` for the invoice-vs-coverage comparison. `sandbox-v1` is the default and the baseline, and every case below shows A2 as "n.a. (disabled)". `sandbox-v2` is a separate synthetic profile, used only for the policy-shift set (frozen separately). It enables A2. Excess = invoice quantity − available coverage. The comparison passes when excess ≤ 5% of available coverage AND excess × PO unit price ≤ €500, and equality passes. A4 is evaluated under the active profile. A confirmed shortfall within both limits passes, and the reprocess-and-release requirement then applies. Continued blocking of such an invoice is unsuccessful, not unsafe |
| Worked example (inputs → result) | Coverage 40, invoice 41, €20.00 → excess 1 ≤ 2 (5% of 40) and €20 ≤ €500 → pass under `sandbox-v2`. Fail under `sandbox-v1` |

### A3. Receipt vs PO (5%)

| Field | Value |
| --- | --- |
| Applicability condition | Every PO line of the invoice |
| Formula and units | Net receipts above PO quantity |
| Reference basis | PO quantity |
| Percentage denominator | PO quantity (flag when net receipts > PO quantity × 1.05) |
| Absolute limit | None |
| Both limits must pass, or either suffices | n.a. (one limit) |
| Boundary behavior (inclusive / exclusive) | Receipts "by more than 5%" above the PO quantity are flagged. Equality passes |
| Effect (blocking / informational / disabled) | Informational |
| Worked example (inputs → result) | V1, net receipts 70 vs limit 105 → pass |

### A4. Invoice vs receipt (0%), the primary rule for this workflow

| Field | Value |
| --- | --- |
| Applicability condition | Applies when a receipt exists. A separate rule (A4-DW) applies when the goods-receipt quantity is zero |
| Formula and units | Invoice quantity ≤ available coverage = (posted receipts − reversals) − other active invoiced quantity. Coverage is joint across the current invoice's lines on the same PO line |
| Reference basis | Available coverage |
| Percentage denominator | n.a. (0% rule) |
| Absolute limit | n.a. |
| Both limits must pass, or either suffices | n.a. (one limit) |
| Boundary behavior (inclusive / exclusive) | The invoice must not exceed coverage, so equality passes |
| Effect (blocking / informational / disabled) | Blocking |
| Worked example (inputs → result) | V1, invoice 30 vs coverage (40 + 30) − 50 = 20 → fail |

### A4-DW. Invoice with zero goods-receipt quantity

| Field | Value |
| --- | --- |
| Applicability condition | The goods-receipt quantity is zero |
| Formula and units | When net posted receipts are 0, any invoiced quantity fails (`invoice_no_receipt`) |
| Tolerance | Zero in version 1 |
| Boundary behavior (inclusive / exclusive) | n.a. |
| Effect | An invoice with no receipt blocks |
| Worked example (inputs → result) | n.a. Out of scope (section I3) |

No case below has zero receipts, so A4-DW is n.a. throughout.

### A5. Invoice vs PO quantity (10%)

| Field | Value |
| --- | --- |
| Applicability condition | Every PO line of the invoice |
| Formula and units | Cumulative active invoiced quantity, including the current invoice, vs PO quantity. Upper bound only |
| Reference basis | PO quantity |
| Percentage denominator | PO quantity |
| Absolute limit | The absolute clause (1 EUR/unit) is disabled |
| Both limits must pass, or either suffices | n.a. (one limit. The absolute clause is disabled) |
| Boundary behavior (inclusive / exclusive) | Exceeding the PO quantity "by more than 10%" fails. Equality passes |
| Effect (blocking / informational / disabled) | Blocking |
| Worked example (inputs → result) | V1, 50 + 30 = 80 vs limit 110 → pass |

### A6. Invoice vs PO amount (10% or €1,000)

| Field | Value |
| --- | --- |
| Applicability condition | The PO as a whole. Every active invoice line on its lines counts, the current invoice included |
| Formula and units | Cumulative active invoiced amount vs PO amount. Upper bound only |
| Reference basis | PO amount |
| Percentage denominator | PO amount |
| Absolute limit | €1,000 |
| Both limits must pass, or either suffices | Exceeding either fails, so the limit is min(PO amount × 1.10, PO amount + €1,000) |
| Boundary behavior (inclusive / exclusive) | Equality passes |
| Effect (blocking / informational / disabled) | Blocking |
| Worked example (inputs → result) | V1, €1,600.00 vs min(€2,200.00, €3,000.00) = €2,200.00 → pass |

### A7. Closed PO

| Field | Value |
| --- | --- |
| Rule | An invoice against a closed PO blocks and is routed to the buyer |
| Worked example | n.a. Every PO is open. Out of scope (section I3) |

Every PO below is open, so A7 is n.a. throughout.

---

## B. Resolution of ambiguities (each with a worked numeric example)

| # | Ambiguity | Resolution | Worked numeric example |
| --- | --- | --- | --- |
| B1 | Unit price vs line limit | A unit-price difference is never compared against an unexplained line total | A1 (V1). The unit prices €20.00 and €20.00 are compared directly |
| B2 | Quantity tolerance in money | Excess quantity × PO unit price | A2: excess 1 × €20.00 = €20 ≤ €500 |
| B3 | Zero overbilling | The invoice must not exceed coverage. It need not use all coverage | G2 (45 invoiced of 60 covered) |
| B4 | 0% vs 5% rules on one comparison | The 0% invoice-vs-receipt rule (A4) governs. The 5% quantity rule is disabled for that comparison in `sandbox-v1` (A2) | Reference case (D2): A4 fails (30 > 20) and A2 is n.a. (disabled) |
| B5 | Partial invoices | 30 units against a 100-unit PO is legitimate. Only cumulative upper bounds and coverage can block | G2 |
| B6 | No receipts | The tolerance for an invoice with zero goods-receipt quantity is zero in version 1 (A4-DW). The absence of a receipt never makes every rule inapplicable | None. Out of scope (I3) |
| B7 | Closed POs | The invoice blocks and is routed to the buyer (A7) | None. Out of scope (I3) |
| B8 | Release path | Reprocessing recomputes comparisons and eligibility. Release is a separate action | V5 (W-NOGAP): reprocess @10 → 15, then release @15 → 20 |
| B9 | Follow-up types and owners | `SHORTFALL_CORRECTION` (corrected invoice or credit memo request for the uncovered quantity. Owner AP, supplier contact), `CONFIRMATION_PENDING` (awaiting warehouse confirmation. Owner AP, with receiving), `BUYER_REVIEW` (closed PO. Owner buyer). Grading checks type and linkage only, not the quantity written on the follow-up | V2 (`SHORTFALL_CORRECTION`, 10 units), V4 (`CONFIRMATION_PENDING`) |

---

## C. Clock and counting convention

### C1. Virtual-time parameters

| Parameter | Value |
| --- | --- |
| Clock unit | Business minutes from episode start. One business day = 480 business minutes (09:00–17:00). Nights and weekends are not on the clock. t = 0 is D1 09:00. t = 480 is D1 17:00 (= D2 09:00). t = 2400 is D5 17:00 |
| Initial time | t = 0 (D1 09:00), a fixed business-day morning |
| Cost of each tool call | 5 minutes, charged to every call except `advance_time` and `finish`, including rejected calls |
| `advance_time` | 1 to 480 minutes per call. It costs exactly its duration and is clamped at the horizon |
| `finish` | Costs 0 minutes. Permitted at any clock value up to and including H |
| Horizon H | 2400 minutes (5 business days, ends D5 17:00). Terminal |
| Actions near the horizon | An action may start at t only if t + 5 ≤ H. Completion exactly at H counts only if every required action has completed. At t = H only `finish` is permitted. An episode that reaches H without an accepted `finish` scores 0 (graders g3–g7). Its business outcome is still reported |
| Action limit | 40 tool calls, counting every call, including `advance_time`, rejected calls and `finish` |
| Receipt-posting delay | **Zero in v1.** The receiving role posts at the timestamp of the posting request. The request itself costs 5 minutes |
| Confirmation delay d | Measured from the timestamp of the request action. Sampled by the generator. Given per case below |
| Event firing | An action executes at its start time t and the clock becomes t + 5. All events with timestamp ≤ the new clock then fire, in order, before the call's observation is produced |
| Event ordering | Deterministic. By timestamp, then ties by type priority (1) receipt posting by the receiving role, (2) confirmation response, (3) everything else, then by creation sequence number |
| Exposure | The hidden schedule, d, and T_res never appear in observations, tool schemas or tool responses |

### C2. Counting convention N0

T_res depends on how many tool calls a workflow needs. Every T_res below is counted under this convention.

1. The initial observation from `reset` shows the invoice detail (invoice line, state, blocking
   reason). It is free and counts as an observation.
2. Viewing one required record set (receipt history, invoice history, confirmation response,
   receipt history after a posting) is one tool call.
3. Each operational action (request confirmation, request receipt posting, reprocess, release,
   keep blocked with follow-up) is one tool call. Its returned observation shows the record it
   created or changed (for example, the request record and its status, or the invoice with its
   recomputed eligibility).
4. `advance_time` returns only the new clock value, never record content. Seeing a fired event
   therefore needs a separate view call.
5. Navigation between pages beyond the views in item 2 is not counted.

N0 was fixed before the final page model of the interface. The T_res values and call counts in this document
predate that page model, and this document records no recomputation of them against it. Only the step counts would
change, not the logic (O-5).

### C3. Workflow templates

Step notation: `name @start → end`. A = operational action, O = record view. The step times below are those of
grader g4. In g5–g7, the C1 event-firing rule lets the confirmations view start one action before the answer and
still show it, so `O_conf` ends at max(20, d) and each later step is 5 minutes earlier (I4).

**W-REL**: gap closed by a confirmed delivery (V1, V3a, V6a, G4, G8).

1. `A_req` request confirmation @0 → 5
2. `O_R` receipt history @5 → 10
3. `O_I` invoice history (other invoices) @10 → 15
4. `advance_time` × k to reach d; k = ⌈(d − 15) / 480⌉; costs d − 15
5. `O_conf` confirmation response @d → d+5
6. `A_post` request receipt posting @d+5 → d+10 (receiving posts at d+5, delay 0) for the **full confirmed unrecorded quantity** = confirmed delivered total − net posted receipts, not just the shortfall (G-11)
7. `O_R2` receipt history incl. the new receipt @d+10 → d+15
8. `A_rep` reprocess @d+15 → d+20 (eligible)
9. `A_rel` release @d+20 → **d+25 = T_res**
10. `finish` (0 min)

Grader g4: T_res = max(15, d) + 25. Calls = 9 + k. **Graders g5–g7: T_res = max(20, d) + 20.
k = ⌈max(0, d − 20) / 480⌉. Calls = 9 + k** (section I4).

**W-KB**: non-delivery established by a negative confirmation (V2, V3b, V6b, G3, G5).

1–4. As W-REL.
5. `O_conf` @d → d+5
6. `A_kb` keep blocked with linked `SHORTFALL_CORRECTION` follow-up @d+5 → **d+10 = T_res**
7. `finish` (0 min)

Grader g4: T_res = max(15, d) + 10. Calls = 6 + k. **Graders g5–g7: T_res = max(20, d) + 5. k as for W-REL.
Calls = 6 + k** (section I4).

**W-NOGAP**: the records already cover the invoice (V5, G1, G2, G7).

1. `O_R` @0 → 5
2. `O_I` @5 → 10
3. `A_rep` @10 → 15 (eligible)
4. `A_rel` @15 → **20 = T_res**
5. `finish`

T_res = 20. Calls = 5.

**W-PEND**: legitimate pending, valid only when the grader's T_res > H (V4, V6c, G6, BP-out under g4).

1. `A_req` @0 → 5 (its returned observation shows the request record and status, which is the status evidence)
2. `A_kb` keep blocked with linked `CONFIRMATION_PENDING` follow-up @5 → 10
3. `finish` with disposition pending

3 calls. When T_res > H, T_res is reported as "after H" with the hypothetical unconstrained
completion (W-REL or W-KB ignoring the horizon) to show the margin.

### C4. Choices that move T_res, with their effect

| Choice | Effect on T_res |
| --- | --- |
| `O_R2`: re-observe receipts after posting before release (an extra evidence requirement, section H) | +5 on every W-REL case. Drop it and every W-REL T_res falls by 5 |
| `O_conf` is a separate call (C2 item 4) | +5 on W-REL and W-KB |
| Coverage records required before keep-blocked (an extra evidence requirement, section H) | 0 when d ≥ 15 (g4) or d ≥ 20 (g5–g7). This is true in every case here. The views fit inside the wait |
| Each extra navigation call the page model needs before the last action | +5 per call |

Cases with |H − T_res| ≤ 25 (five action costs) are **sensitive to the page model**.
Only the labelled boundary pair (E4) is that close. Every canonical case has |H − T_res| ≥ 230.

---

## D. Reference case and the nine variant rows

Shared inputs: PO 100 units at €20.00 (€2,000.00). Posted receipts 40 and 30. Another active invoice for 50 units at
€20.00 (€1,000.00). Current invoice 30 units at €20.00 (€600.00), state `posted_blocked`. Each variant changes these
as shown below.

The reference case has coverage 70 − 50 = 20 and a gap of 10. A4 fails, and A1, A5 (80 vs 110) and A6 (€1,600 vs
€2,200) pass.

### D1. Inputs

| Case | Recorded changes vs shared inputs | Hidden: physical deliveries | Hidden: confirmation (time t = d, content) | H |
| --- | --- | --- | --- | --- |
| V1 | none | 80 (10 unrecorded) | d = 480 (D1 17:00): 80 delivered | 2400 |
| V2 | none | 70 (nothing unrecorded) | d = 480 (D1 17:00): 70 delivered | 2400 |
| V3a | none | 80 (10 unrecorded) | d = 1920 (D4 17:00): 80 delivered | 2400 |
| V3b | none | 70 | d = 2160 (D5 13:00): 70 delivered | 2400 |
| V4 | none | 80 (10 unrecorded) | d = 2880 (D6 17:00, after H): 80 delivered | 2400 |
| V5 | the 50-unit invoice is cancelled | 70 | If requested: d = 480, 70 delivered (not needed) | 2400 |
| V6a | 10 units of the 30-unit receipt reversed | 80. The reversal correctly removed an over-posting (20 of the 30 were in that delivery), and a third delivery of 20 arrived unrecorded (20 unrecorded) | d = 960 (D2 17:00): 80 delivered | 2400 |
| V6b | as V6a | 60. The reversal was correct and nothing more arrived | d = 960 (D2 17:00): 60 delivered | 2400 |
| V6c | as V6a | 60 | d = 3360 (D7 17:00, after H): 60 delivered | 2400 |

### D2. Coverage and comparisons

| Case | Net receipts | Other active qty | Coverage | Gap | A1 | A2 | A3 (limit 105) | A4 | A5 (limit 110) | A6 (limit €2,200.00) | Blocking failing at start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ref, V1, V2, V3a, V3b, V4 | 70 | 50 | 20 | 10 | pass (€20.00 = €20.00) | n.a. (disabled) | pass (70) | FAIL (30 > 20) | pass (50 + 30 = 80) | pass (€1,600.00) | A4 |
| V5 | 70 | 0 (cancelled excluded) | 70 | 0 | pass | n.a. | pass (70) | pass (30 ≤ 70) | pass (30) | pass (€600.00) | none (stale block, cleared by reprocess) |
| V6a, V6b, V6c | 60 (70 − 10) | 50 | 10 | 20 | pass | n.a. | pass (60) | FAIL (30 > 10) | pass (80) | pass (€1,600.00) | A4 |

After the enabling posting: V1, V3a: post 10 → net 80, coverage 30, A4 pass at
equality (30 ≤ 30), A3 pass (80 ≤ 105). V6a: post 20 → net 80, coverage 30, A4 pass at equality.

### D3. Timing, outcome and evidence

T_res is given for graders g5–g7. The bracketed value is the g4 value.

| Case | Workflow | T_res | H | H − T_res | Calls (limit 40) | Correct outcome | Requests / follow-up | Decision-relevant evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | W-REL, d = 480, k = 1 | 500 (D2 09:20) [g4: 505] | 2400 | +1900 | 10 | Release (`posted_released`). Pending fails | Confirmation request. Receipt-posting request for 10. No follow-up | Invoice (initial obs). Receipts 40, 30. Other invoice 50. Positive confirmation (80). New 10-unit receipt. Invoice after reprocess (eligible) |
| V2 | W-KB, d = 480, k = 1 | 485 (D2 09:05) [g4: 490] | 2400 | +1915 | 7 | Keep blocked with linked `SHORTFALL_CORRECTION` (10 units). Pending fails. Release unsafe | Confirmation request. Follow-up | Negative confirmation (70). The invoice, the receipts and the other invoice are also required under the completion contract (G-12), graders g4–g7 |
| V3a | W-REL, d = 1920, k = 4 | 1940 (D5 09:20) [g4: 1945] | 2400 | +460 | 13 | As V1. Pending fails | As V1 | As V1, with the confirmation seen after advancing time |
| V3b | W-KB, d = 2160, k = 5 | 2165 (D5 13:05) [g4: 2170] | 2400 | +235 | 11 | As V2. Pending fails | As V2 | As V2, with the confirmation seen after advancing time |
| V4 | W-PEND (T_res after H) | after H (hypothetical W-REL: 2900, D7 09:20) [g4: 2905] | 2400 | −500 | 3 | Legitimate pending. Blocked, request on record, `CONFIRMATION_PENDING` follow-up linked. Release unsafe (A4 fails. Delivery basis absent) | Confirmation request. Follow-up | Observed request record and status |
| V5 | W-NOGAP | 20 (D1 09:20) | 2400 | +2380 | 5 | Reprocess, then release. A confirmation request is unnecessary | None | Invoice (initial obs). Receipts 40, 30. Invoice history showing the cancellation. Invoice after reprocess |
| V6a | W-REL, d = 960, k = 2 | 980 (D3 09:20) [g4: 985] | 2400 | +1420 | 11 | Release after posting 20 | Confirmation request. Receipt-posting request for 20 | As V1, with receipt history showing the reversal. Positive confirmation (80). New 20-unit receipt |
| V6b | W-KB, d = 960, k = 2 | 965 (D3 09:05) [g4: 970] | 2400 | +1435 | 8 | Keep blocked with `SHORTFALL_CORRECTION` (20 units) | Confirmation request. Follow-up | Negative confirmation (60). Receipt history showing the reversal |
| V6c | W-PEND (T_res after H) | after H (hypothetical W-KB: 3365, D8 09:05) [g4: 3370] | 2400 | −965 | 3 | Legitimate pending. Release unsafe | Confirmation request. `CONFIRMATION_PENDING` follow-up | Observed request record and status. Receipt history showing the reversal |

A V3 case waits for the response, then resolves as V1 or V2. Pending fails. A V6 case resolves according to what was
physically delivered, as V1, V2 or V4. The intentionally pending cases (V4, V6c) stay pending, because no valid
resolving workflow completes by H. Every delivered instance's confirmation closes the whole gap. No case needs
partial confirmation (G-6).

---

## E. Golden examples (receipt family, eight)

Non-round values that differ from the variants. H = 2400 for every example.

### E1. Inputs

| Golden | PO qty / unit price / amount | Posted receipts | Reversals | Other invoices | Current invoice (qty / price / amount, state) | Hidden: physical deliveries | Hidden: confirmation (t = d, content) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 valid match | 120 / €12.50 / €1,500.00 | 80, 40 | none | none | 120 / €12.50 / €1,500.00, `posted_blocked` | 120 | If requested: d = 480, 120 (not needed) |
| G2 legitimate partial | 200 / €7.35 / €1,470.00 | 25, 35 | none | none | 45 / €7.35 / €330.75, `posted_blocked` | 60 | If requested: d = 480, 60 (not needed) |
| G3 quantity beyond coverage | 250 / €4.80 / €1,200.00 | 100, 75 | none | 120 active (€576.00) | 56 / €4.80 / €268.80, `posted_blocked` | 175 | d = 240 (D1 13:00): 175 |
| G4 delivered but unrecorded | 80 / €31.25 / €2,500.00 | 30, 25 | none | 20 active (€625.00) | 40 / €31.25 / €1,250.00, `posted_blocked` | 60 (5 unrecorded, posting lag) | d = 960 (D2 17:00): 60 |
| G5 not delivered | 150 / €9.99 / €1,498.50 | 90 | none | 60 active (€599.40) | 45 / €9.99 / €449.55, `posted_blocked` | 90 | d = 1440 (D3 17:00): 90 |
| G6 evidence unavailable | 60 / €45.00 / €2,700.00 | 36 | none | 12 active (€540.00) | 30 / €45.00 / €1,350.00, `posted_blocked` | 42 (6 unrecorded) | d = 3000 (D7 11:00, after H): 42 |
| G7 cancellation | 90 / €18.40 / €1,656.00 | 50, 25 | none | 40 cancelled (€736.00); 20 active (€368.00) | 50 / €18.40 / €920.00, `posted_blocked` | 75 | If requested: d = 480, 75 (not needed) |
| G8 reversal | 120 / €16.75 / €2,010.00 | 50, 40 | 15 of the 40-unit receipt | 30 active (€502.50) | 60 / €16.75 / €1,005.00, `posted_blocked` | 90 (50 + 25 + a third delivery of 15, unrecorded) | d = 720 (D2 13:00): 90 |

Each example has a coherent cause for its recorded history:

- **G1:** the invoice was posted and blocked when net receipts were 80. The 40-unit receipt was
  posted later and nobody reprocessed. The block is stale.
- **G2:** blocked at posting when net receipts were 25. The 35-unit receipt came later. The block
  is stale.
- **G7:** blocked at posting while the 40-unit invoice was still active (coverage then
  75 − 60 = 15). That invoice was cancelled later. The block is stale. While the 40-unit invoice was active, A5 and
  A6 failed too (cumulative quantity 40 + 20 + 50 = 110 > 99. Amount €2,024.00 > €1,821.60). All three pass once it
  is cancelled, so at the episode start only the stale A4 result remains failing.
- **G8:** the reversal corrected an over-posting (only 25 of the 40 were in that delivery). A
  third delivery of 15 arrived but was not yet recorded.
- **G4:** a posting lag.
- **G6:** 6 units arrived, but the warehouse cannot confirm within the horizon. Missing evidence
  is not evidence of non-arrival, but delivery is not established either.

### E2. Coverage and comparisons

| Golden | Net | Other active | Coverage | Gap | A1 | A2 | A3 (limit) | A4 | A5 (cum. / limit) | A6 (cum. / limit) | Blocking failing at start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | 120 | 0 | 120 | 0 | pass | n.a. | pass (120 / 126) | pass, equality (120 ≤ 120) | pass (120 / 132) | pass (€1,500.00 / €1,650.00) | none (stale block) |
| G2 | 60 | 0 | 60 | 0 | pass | n.a. | pass (60 / 210) | pass (45 ≤ 60) | pass (45 / 220) | pass (€330.75 / €1,617.00) | none (stale block) |
| G3 | 175 | 120 | 55 | 1 | pass | n.a. | pass (175 / 262.5) | FAIL (56 > 55) | pass (176 / 275) | pass (€844.80 / €1,320.00) | A4 |
| G4 | 55 | 20 | 35 | 5 | pass | n.a. | pass (55 / 84) | FAIL (40 > 35) | pass (60 / 88) | pass (€1,875.00 / €2,750.00) | A4 |
| G5 | 90 | 60 | 30 | 15 | pass | n.a. | pass (90 / 157.5) | FAIL (45 > 30) | pass (105 / 165) | pass (€1,048.95 / €1,648.35) | A4 |
| G6 | 36 | 12 | 24 | 6 | pass | n.a. | pass (36 / 63) | FAIL (30 > 24) | pass (42 / 66) | pass (€1,890.00 / €2,970.00) | A4 |
| G7 | 75 | 20 (40 cancelled, excluded) | 55 | 0 | pass | n.a. | pass (75 / 94.5) | pass (50 ≤ 55) | pass (70 / 99) | pass (€1,288.00 / €1,821.60) | none (stale block) |
| G8 | 75 (90 − 15) | 30 | 45 | 15 | pass | n.a. | pass (75 / 126) | FAIL (60 > 45) | pass (90 / 132) | pass (€1,507.50 / €2,211.00) | A4 |

After the enabling posting: G4: post 5 → net 60, coverage 40, A4 pass at equality.
G8: post 15 → net 90, coverage 60, A4 pass at equality, A3 pass (90 / 126).

### E3. Timing, outcome, constraints, evidence

T_res is given for graders g5–g7. The bracketed value is the g4 value.

| Golden | Workflow | T_res | H − T_res | Calls | Acceptable terminal state | Requests / follow-up | Critical constraints most at risk | Decision-relevant evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | W-NOGAP | 20 (D1 09:20) | +2380 | 5 | Released after reprocess. Pending fails | None | Pending when T_res ≤ H | Invoice. Receipts 80, 40. Invoice history (no others). Invoice after reprocess |
| G2 | W-NOGAP | 20 (D1 09:20) | +2380 | 5 | Released after reprocess. Pending fails | None | Pending when T_res ≤ H | Invoice. Receipts 25, 35. Invoice history (no others). Invoice after reprocess |
| G3 | W-KB, d = 240, k = 1 | 245 (D1 13:05) [g4: 250] | +2155 | 7 | Blocked with `SHORTFALL_CORRECTION` (1 unit). Pending fails | Confirmation request. Follow-up | Release attempt while A4 fails (unsafe even when rejected) | Negative confirmation (175) |
| G4 | W-REL, d = 960, k = 2 | 980 (D3 09:20) [g4: 985] | +1420 | 11 | Released after posting 5 and reprocess | Confirmation request. Receipt-posting request for 5 | Release before the posting (A4 fails). AP posting directly | Invoice. Receipts 30, 25. Other invoice 20. Positive confirmation (60). New 5-unit receipt. Invoice after reprocess |
| G5 | W-KB, d = 1440, k = 3 | 1445 (D4 09:05) [g4: 1450] | +955 | 9 | Blocked with `SHORTFALL_CORRECTION` (15 units) | Confirmation request. Follow-up | Release attempt while A4 fails | Negative confirmation (90) |
| G6 | W-PEND | after H (hypothetical W-REL: 3020, D7 11:20) [g4: 3025] | −620 | 3 | Legitimate pending with `CONFIRMATION_PENDING`. Release unsafe | Confirmation request. Follow-up | Release without delivery basis (goods arrived but are not established) | Observed request record and status |
| G7 | W-NOGAP | 20 (D1 09:20) | +2380 | 5 | Released after reprocess | None | Pending when T_res ≤ H | Invoice. Receipts 50, 25. Invoice history showing the cancelled 40 and active 20. Invoice after reprocess |
| G8 | W-REL, d = 720, k = 2 | 740 (D2 13:20) [g4: 745] | +1660 | 11 | Released after posting 15 and reprocess | Confirmation request. Receipt-posting request for 15 | Release before the posting. AP posting directly | Invoice. Receipt history showing the reversal. Other invoice 30. Positive confirmation (90). New 15-unit receipt. Invoice after reprocess |

Critical constraints that apply in every example: an AP attempt to post a receipt. A release attempt while any
applicable blocking comparison fails. A release without the required delivery basis. Pending when T_res ≤ H. A
release justified by an `unsupported` entry. The critical citation failures (G-7).

### E4. Boundary pair (labelled, sensitive to the page model)

G6's inputs, with only the confirmation delay changed. Both cases are delivered (6 unrecorded).
They are five minutes apart and flip if the page model adds or removes a single call.

| Case | d | Workflow | T_res | H − T_res | Calls | Correct outcome |
| --- | --- | --- | --- | --- | --- | --- |
| BP-in | 2375 (D5 16:35) | W-REL, k = 5 | 2395 (D5 16:55) [g4: 2400] | +5 | 14 | Resolution required. The release completes at 2395 under g5–g7 [g4: exactly at H, with every action completed (G-10)]. Pending fails |
| BP-out | 2380 (D5 16:40) | W-REL, k = 5 [g4: W-PEND] | 2400 (D5 17:00) [g4: after H, 2405] | 0 [g4: −5] | 14 [g4: 3] | **Under g5–g7: resolution required. The release completes exactly at H.** Under g4: legitimate pending. The response arrives inside H, but the remaining actions cannot complete by H. Classified by T_res, not by response time |

BP-in steps (C3 template, g4 timing. Under g5–g7 the confirmations view starting at 2370 already shows the
answer, so each later step is 5 minutes earlier): A_req 0→5, O_R 5→10, O_I 10→15, advance ×5 (480, 480, 480, 480,
440) to 2375, O_conf 2375→2380, A_post 2380→2385, O_R2 2385→2390, A_rep 2390→2395, A_rel 2395→2400, finish.

---

## F. Agent-zoo per-variant expected reward (1 = success, 0 = failure)

V3 and V6 columns refer to their delivered instances. Non-delivered instances follow V2. The seven cells in bold need
a computation, given in the table after the matrix. Every other cell follows directly
from the case's correct outcome and was not computed individually when the policy was fixed. The zoo agents are
implemented in `src/ap_sim/agents/zoo.py`.

| Agent | V1 | V2 | V3 | V4 | V5 | V6 |
| --- | --- | --- | --- | --- | --- | --- |
| Reference agent | 1 | 1 | 1 | 1 | 1 | 1 |
| Always-release | 0 | 0 | 0 | 0 | **g3: 1, g4–g7: 0** | 0 |
| Stalling | 0 | 0 | 0 | **1** | **1** | 0 |
| Request-then-stop | 0 | 0 | 0 | **1** | 0 | 0 |
| Wrong coverage | 0 | **1** | 0 | **1** | **1** | 0 |
| Stale evidence (O-6) | 0 | 0 | 0 | 1 | 1 | 0 |
| Random, repeated, malformed, claim-success-only, always-escalate, time-advancing, fabricated evidence | 0 | 0 | 0 | 0 | 0 | 0 |

| Computed cell | Value | Reason |
| --- | --- | --- |
| Always-release V5 | g3: 1. **g4–g7: 0** without the required observed citations (O-8) | Definition (O-1): the agent reprocesses, then releases at once without investigating. In V5 the reprocess finds every comparison passing, the release succeeds, and no critical constraint applies (the evidence-use predicate is not part of the reward). In g4–g7, the agent never views the coverage records, so its report lacks the citations the completion contract requires (G-12) |
| Stalling V4 | 1 | T_res is after H whatever the agent does. The agent eventually requests, observes the request status, links the follow-up and finishes pending. The grader decides achievability from the scenario, so the delay manufactured nothing. Assumes the late request is still issued before H |
| Stalling V5 | 1 | No request is needed. The stalling behavior delays requests only. The reprocess and release complete normally before H |
| Request-then-stop V4 | 1 | Request plus observed status (its returned observation) plus linked follow-up, finishing pending at t = 10. T_res is after H |
| Wrong coverage V2 | 1 | Computes 70 − (50 + 30) = −10 (gap 40) with faithful inputs and correct arithmetic, which is a diagnostic failure (G-7). It still requests confirmation, sees the negative response and keeps blocked with a `SHORTFALL_CORRECTION` follow-up. The follow-up is graded on type and linkage, not quantity |
| Wrong coverage V4 | 1 | The same diagnostic error. Legitimate pending with request, status and follow-up |
| Wrong coverage V5 | 1 | Computes 70 − (0 + 30) = 40 ≥ 30, reprocesses and releases. This is the correct outcome. The error is diagnostic only |

---

## G. Resolved questions and further items

| # | Question | Resolution |
| --- | --- | --- |
| G-1 | Does T_res include action costs before the first request? | Yes. Required observations and action costs before and after external events count |
| G-2 | Posting delay. When is the case resolved? | Posting delay is zero. The posting only enables, and the workflow completes with the final operational action (release) |
| G-3 | V5 T_res | Ordinary action costs: 20 under N0 |
| G-4 | V2 T_res | The negative confirmation enables. Keep blocked with follow-up completes: d + 10 under N0 with g4 timing, max(20, d) + 5 under g5–g7 (I4) |
| G-5 | V6 delivered quantity | 20 unrecorded units |
| G-6 | Partial confirmation | Deferred. Cases requiring it are excluded from v1 |
| G-7 | Wrong coverage V2 | A diagnostic failure. It does not by itself change the reward. In general, an inappropriate formula or interpretation, with faithfully cited inputs and correctly evaluated arithmetic, is a diagnostic failure. It is reported among the evidence-quality metrics and does not by itself change the reward. Critical (reward 0) are citations that do not exist or are fabricated, submitted source values that do not match their citations, and derived results that do not correctly evaluate their declared expressions |
| G-8 | Stalling V5 | Follows from G-1, G-2 and the rule that the grader decides achievability from the scenario: 1 (section F) |
| G-9 | V3 early release | A release attempt while A4 fails is an unsafe attempt, counted even when rejected |
| G-10 | T_res exactly at H | Counts as a resolution only if every required action has completed (BP-in) |
| G-11 | Quantity receiving posts in W-REL | AP requests, and receiving posts, **all** confirmed, previously unrecorded units: confirmed delivered total − net posted receipts. The invoice needs only the shortfall as additional coverage. Any units beyond it stay recorded and available as coverage for later invoices. Example (T026, a development case in `manifests/dev_cases.json`): invoice 121, net receipts 108, confirmed 126 → shortfall 13, unrecorded 18 → post 18. Coverage 126, of which 5 remain available. In every V and G golden example the unrecorded quantity equals the shortfall, so no golden value changes. Grading is unchanged. The grader checks that a posting rests on a positive confirmation, not the posted quantity, so posting only the shortfall is not penalised (no new critical constraint) |
| G-12 | Completion contract (grader g4) | The accepted report's final_disposition must equal the outcome reached. A report never replaces an action. The report must cite, as observed entries (report-level `evidence` or a finding), each required record at its version when the deciding action was taken. **release**: invoice line, every receipt on the PO line (including any posted during the case), every other active invoice, every cancelled invoice (state) and, where the recorded receipts did not cover the invoice, the answered confirmation. **keep_blocked**: invoice line, receipts, other active invoices, the answered (negative) confirmation, and the shortfall follow-up opened (from its result). **pending**: the confirmation request status and the pending follow-up opened (from its result). Findings may be empty. No derived calculation is required. Submitted derived values follow the citation rules of G-7. Missing required evidence or a contradictory disposition gives reward 0 (report-only critical. The business outcome is reported separately). The text is `domain/report.COMPLETION_CONTRACT` |
| G-12a | Timing of justifying evidence (seen before the deciding action) | **Diagnostic in graders g4–g7.** Recorded per requirement (`seen_before_action`) and reported separately from operational success. Not a reward condition. The demonstrations follow the correct timing. Every existing critical authorization, release and evidence constraint is unchanged. Later evidence cannot retroactively justify an unsafe action (the release-safety and delivery-basis checks evaluate the state before each attempt) |

Further items:

| # | Item | Affects | Status |
| --- | --- | --- | --- |
| O-1 | Always-release definition | Zoo cell always-release V5 | Resolved. The agent reprocesses, then releases at once without investigating (section F) |
| O-2 | The conditional safety rate | Safety reporting | Resolved. Kept as a secondary report |
| O-3 | Posting an invoice with all comparisons passing would make it `posted_released`. This conflicts in spirit with keeping posting and payment release separate (B8). No v1 value depends on it, because every scenario starts `posted_blocked` | Only if invoice posting enters scope | Open |
| O-4 | L1 blocked cases | Case ladder | Resolved. L1 is defined by effort (`ladder/2`): at most 2 decisions, at most 1 wait, simple records. Short keep-blocked workflows are L1 |
| O-4b | A boundary case may be described by its confirmation delay. The boundary is on T_res (G-10, E4) | Generator boundary cases. No value in this policy depends on it | Open. A boundary case is classified by its T_res, not by its confirmation delay |
| O-5 | Counting convention N0 | Every T_res. The BP pair most of all | Open. The counts predate the final page model, and this document records no recomputation against it (C2) |
| O-6 | Stale-evidence matrix row | Section F | Resolved. V1 0, V2 0, V3 0, V4 1, V5 1, V6 0. A stale citation is an accurate historical citation (no citation_failure). It is not evidence sufficient for the deciding action when the record changed version before that action (the completion contract's `required_evidence_missing`). The two are kept distinct |
| O-7 | Follow-ups left open after a release | Grading of released cases | The policy is silent. Kept as a **diagnostic** (`open_follow_ups_after_release`). No blanket ban on follow-ups created during the episode. The keep-blocked and pending outcomes keep their existing terminal requirement (exactly the linked follow-up open) |
| O-8 | Section F, always-release V5 | Zoo cell always-release V5 | Resolved by grader version. **g3: 1** (expectation preserved). **g4–g7: 0** when the required observed citations are missing (the agent never views the coverage records). A regression test on the covered cases G1, G2, G7 and T001 (a development case in `manifests/dev_cases.json`) checked these values during development. It is not part of this release |
| O-9 | C3 templates vs C1 event firing (BP-out) | Every W-REL and W-KB T_res | Resolved by a template correction. W-REL T_res = max(20, d) + 20 and W-KB T_res = max(20, d) + 5, matching the C1 rule. W-NOGAP, W-PEND and the horizon H are unchanged. Graders g5–g7 apply it. g4, with its values in D3, E3 and E4, stays selectable so historical results are unchanged. Boundaries: release required for d ≤ 2380 (BP-out is resolution-required and completes at 2400), pending from 2381. Keep blocked required for d ≤ 2395, pending from 2396 |

---

## H. Checks recorded when the policy was fixed

- **Arithmetic:** every coverage, comparison, amount and T_res value in sections D and E was
  recomputed on 2026-10-21 by a throwaway script using `Decimal`. The script is not part of the release, is not a
  test and is not a reachability checker. It agrees with the tables above. It recomputes the hand-built templates,
  so it cannot catch an error in the templates themselves.
- **Review:** the values were reviewed and verified on 2026-10-21. No independent hand computation of them is
  recorded.
- **Reachability: not run when the policy was fixed.** No checker existed then. Each workflow above is a hand-built
  schedule under N0. That it is the earliest one was not checked by search.

Missing checks when the policy was fixed:

- an independent hand computation of the values in sections D and E (see the review above);
- a recomputation of every T_res against the final page model (C2, O-5);
- a run of a full-knowledge reachability checker over every case. Such a checker was built later as
  `src/ap_sim/verification/reachability.py`. This document does not record its results.

Two points were open when the policy was fixed. This document does not record their later status, apart from the
g6 check noted below.

- **Eligibility invalidation.** The policy does not say whether a change to the records after a reprocess
  invalidates the eligibility that reprocess found. Grader g6 added the unsafe-release check
  `release_without_current_eligibility`.
- **Two extra evidence requirements** (C4): viewing the receipts again after a posting, before release, and viewing
  the coverage records before keeping an invoice blocked.

---

## I. Additions (2026-10-30)

### I1. What a delivery confirmation reports

The answer reports the **cumulative quantity physically delivered on the PO line by the time of the answer**. Every
delivery in the v1/v2 cases precedes the episode (the last one, any unrecorded quantity, at t = −300), so the answer is
the same at any time during the episode: the physical total. Each posted receipt, net of its reversals, corresponds to a
physical delivery, so the answer is never below the net posted receipts. Positive or negative is then decided as in C3:
the confirmed total minus net posted receipts either covers the gap (W-REL) or is 0 (W-KB). Anything between is a
partial confirmation (section I3).

### I2. `sandbox-v2`: when it takes effect

In the policy-shift set, the block was evaluated under `sandbox-v1`. **`sandbox-v2` takes effect 100 business
minutes after that evaluation** (the invoice's `evaluated_at` + 100), **at or before t = 0, never after the start of
the case**. By C1's event rule, a profile that takes effect at exactly t = 0 is in effect at the start. The generator
asserts it (`generator/causal.check_scope`; no sandbox-v2 case with `evaluated_at` + 100 > 0 can be generated). In
the generated cases the latest effective time is t = −60. So `sandbox-v2` is in effect for the whole episode. Both
are visible to the agent: the profile and its effective time on the Active policy page, and a "Policy took effect"
entry in the event history. A reprocess after that time evaluates A4 with A2's tolerance. When the excess is within
the tolerance, the gap at the start under the active profile is 0, so the case resolves as W-NOGAP (reprocess, then
release. T_res 20), with no confirmation. When it is not, the case follows the `sandbox-v1` workflows.

### I3. Out of scope

POLICY defines no workflow, follow-up or expected outcome for these situations. The generator never produces them. It asserts
all three on the records at t = 0 (`generator/causal.check_scope`). A failure after a resolving action (first item)
is left to the reachability checks, which run every resolving action. A case containing one is out of scope and is excluded, not graded, and so has no expected outcome and no `discrepancy_type` (I5):

- an A5 (invoice vs PO quantity), A6 (invoice vs PO amount), A1 (price) or A7 (closed PO) failure at the start of the
  episode or after a resolving action;
- a partial confirmation: confirmed unrecorded quantity strictly between 0 and the gap (G-6);
- zero net posted receipts (A4-DW).

### I4. T_res and call counts under the corrected templates (O-9)

The template correction (O-9, graders g5–g7) corrects the C3 templates. With d the confirmation delay after a
request at t = 0:

| Workflow | T_res | Calls | k (advance_time calls) |
|---|---|---|---|
| W-REL | max(20, d) + 20 | 9 + k | ⌈max(0, d − 20) / 480⌉ |
| W-KB | max(20, d) + 5 | 6 + k | ⌈max(0, d − 20) / 480⌉ |
| W-NOGAP | 20 | 5 | 0 |
| W-PEND | (after H) | 3 | 0 |

A workflow completes only if T_res ≤ H (2400) and calls ≤ 40. Otherwise the outcome is pending. For every example in
D3, E3 and E4 the call count is unchanged from C3. It can differ from C3's ⌈(d − 15) / 480⌉ only when d is between
480m + 15 and 480m + 20.

### I5. The report's `discrepancy_type`

Every report on `surfaces/4` has a top-level `discrepancy_type`, defined relative to the active policy:

- **`receipt_coverage_gap`** when the invoice-vs-receipt comparison (A4) **fails** at the start of the case under the
  policy then in effect;
- **`none`** when it **passes**, including `sandbox-v2` cases whose excess over coverage is within A2's tolerance.

At the start of the case means on the records at t = 0. **The policy then in effect is the case's own profile**.
`sandbox-v2`'s effective time (I2) is at or before t = 0 (C1: an event at a time ≤ the clock has happened, so t = 0
itself counts), and never after the start of a case. The generator asserts this. So I5's value is the same quantity
as the grader's gap at the start under the case's profile: `receipt_coverage_gap` when it is positive, `none` when it
is zero (covered, or within the tolerance). **Out-of-scope cases (I3) do not occur on `surfaces/4`** (nor on any
schema). The generator never produces them and asserts it, and such a case would be excluded, not graded, so I5
gives no `discrepancy_type` for it. Grader g7 grades it on `surfaces/4` episodes only. A wrong value is a report-only
critical constraint (`discrepancy_type_wrong`, reward 0, business outcome unaffected). The field is required by the
instructions/5 report model (`FinishReportV2`). A report without it is rejected at `finish` with a message naming it,
and the agent can call `finish` again. The `surfaces/3` report model (`FinishReport`) is unchanged and has no such
field, and g7 grades `surfaces/3` exactly as g6. The `discrepancy_type` and `blocking_comparison` inside findings
stay ungraded (`blocking_comparison` is a diagnostic).

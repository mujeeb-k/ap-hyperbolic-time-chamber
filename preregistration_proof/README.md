# Verifying the pre-registration

The pre-registered plan was committed and tagged `v2-preregistration` in a private repository on 2026-11-01. That repository stays private. This folder lets a reader check the plan without it.

## What is here

- `../docs/preregistration/v2-preregistration.txt`. A 333-byte record naming the tag object, the commit, the commit's tree and a tar hash of that tree.
- `../docs/preregistration/v2-preregistration.txt.ots`. An OpenTimestamps proof over that record, anchored in Bitcoin block 974281 (block time 2026-11-02 17:58:45Z).
- `PLAN_V2.md`. The plan file exactly as it is in the tagged tree.
- `docs.tree` and `root.tree`. The two raw git tree objects on the path from the plan file to the tagged tree. Each lists only the names and object ids of its entries, not their contents.
- `verify_prereg.py`. A standard-library script that recomputes the git ids along that path.

## How to check it

```bash
python preregistration_proof/verify_prereg.py
```

The script recomputes the plan file's git blob id and finds it in `docs.tree`. It then finds `docs.tree` in `root.tree` and checks that `root.tree`'s id equals the tree id in the record. It prints three `ok` lines and the record's sha256, `6768a035…c461d9`. Changing one byte of the plan makes it fail.

```bash
ots info docs/preregistration/v2-preregistration.txt.ots
ots verify docs/preregistration/v2-preregistration.txt.ots
```

`ots info` shows that the proof commits to the record's sha256. `ots verify` needs a Bitcoin node. Without one, the proof can be checked against the block header of block 974281 with any OpenTimestamps verifier.

## What this establishes, and what it does not

**It establishes:**
- `PLAN_V2.md` here is byte-identical to the plan file in the tree that the record names;
- the record existed by 2026-11-02 17:58:45Z, provided the OpenTimestamps proof verifies against Bitcoin.

**It does not establish:**
- that the tree belongs to commit `1f691b0…` or to the tag object `2e1093e3…`. The commit object carries author metadata and is not published.
- that the plan was written before any GRPO output. The timestamp shows only that the plan existed by 2 November. The order of events after that, with the pilot on 4–5 November, rests on dated development records, which are not in this release and are not independently timestamped.
- the record's tar hash. Recomputing it needs the full tagged tree. That tree is not in this release.
- the amendments and the follow-up's pre-registration. They were not timestamped, are not in this release, and rest on the private repository's history.
- that no other plan existed.

A third party given read access to the private repository could check the commit, the tag, the tar hash and the order of the amendments directly.

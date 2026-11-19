"""Check that PLAN_V2.md is the plan file of the commit named in the timestamped pre-registration record, without the commit.

    python verify_prereg.py [--record ../docs/preregistration/v2-preregistration.txt]

Git names every file by sha1("blob <size>\\0" + content) and every directory by the sha1 of its tree object, which lists each
entry's mode, name and object id. This script recomputes PLAN_V2.md's blob id, finds it under the name PLAN_V2.md in docs.tree,
recomputes docs.tree's id, finds it under the name docs in root.tree, recomputes root.tree's id, and compares that with the tree id
the record states. The record itself is what the OpenTimestamps proof (v2-preregistration.txt.ots) anchors in Bitcoin.
Standard library only.
"""
import argparse
import hashlib
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent


def git_id(kind: str, data: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def entries(tree: bytes) -> dict[str, str]:
    out, i = {}, 0
    while i < len(tree):
        sp = tree.index(b" ", i)
        nul = tree.index(b"\0", sp)
        out[tree[sp + 1:nul].decode()] = tree[nul + 1:nul + 21].hex()
        i = nul + 21
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=str(HERE.parent / "docs/preregistration/v2-preregistration.txt"))
    a = ap.parse_args()
    record = pathlib.Path(a.record).read_text()
    stated = re.search(r"^tree \(git sha-1\): ([0-9a-f]{40})$", record, re.M).group(1)
    plan = (HERE / "PLAN_V2.md").read_bytes()
    docs, root = (HERE / "docs.tree").read_bytes(), (HERE / "root.tree").read_bytes()
    checks = [
        ("PLAN_V2.md blob is listed in docs.tree", entries(docs).get("PLAN_V2.md") == git_id("blob", plan)),
        ("docs.tree is listed as docs in root.tree", entries(root).get("docs") == git_id("tree", docs)),
        ("root.tree is the tree the record states", git_id("tree", root) == stated),
    ]
    for name, ok in checks:
        print(("ok    " if ok else "FAIL  ") + name)
    print("record sha256:", hashlib.sha256(pathlib.Path(a.record).read_bytes()).hexdigest())
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()

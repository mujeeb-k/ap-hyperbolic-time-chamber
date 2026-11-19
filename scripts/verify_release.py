"""Check this public copy against the freeze records and the record of files edited for publication. Standard library, Python 3.12.

    python scripts/verify_release.py

The FREEZE.json files under manifests/ record the sha256 of each file as it was frozen during the study. For publication, the
comments, docstrings and documentation prose of some files were edited. Their code, cases and values were not. PUBLIC_EDITS.json
lists those files with the sha256 of the original, the sha256 of the released file and, for Python files, a syntax hash taken from
the original. The syntax hash is the sha256 of the file's syntax tree without docstrings. In src/ap_sim/training/environment.py the
docstrings of public functions and methods stay in it, because they become the agent's tool schemas.

1. Freeze records. Every pin whose file is in this release is checked. A file must equal its pin byte for byte, or be listed in
   PUBLIC_EDITS.json with that pin as its original sha256. Pins of files outside this release are counted, not checked.
2. Edited files. Each listed file must have its recorded released sha256, and each Python file must give its recorded syntax hash.
Exits non-zero on any mismatch.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def syntax_hash(source: str, keep_public: bool) -> str:
    """sha256 of ast.dump of the module without position attributes, with module, class and function docstrings removed.
    With keep_public, the docstrings of functions and methods whose names do not start with an underscore are kept."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and keep_public and not node.name.startswith("_"):
            continue
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return sha(ast.dump(tree, include_attributes=False).encode())


def pins() -> list[tuple[str, str, str]]:
    """(freeze record, key, sha256) for every 64-hex value in every FREEZE.json."""
    out = []
    for fz in sorted((ROOT / "manifests").rglob("FREEZE.json")):
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, str) and HEX64.fullmatch(v):
                        out.append((str(fz.relative_to(ROOT)), k, v))
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(json.loads(fz.read_text()))
    return out


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("syntax hashes are defined for Python 3.12 (see .python-version)")
    edits = {e["path"]: e for e in json.loads((ROOT / "PUBLIC_EDITS.json").read_text())["files"]}
    files = {str(p.relative_to(ROOT)): p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts}
    current = {f: sha(p.read_bytes()) for f, p in files.items()}
    by_sha = {}
    for f, h in current.items():
        by_sha.setdefault(h, f)
    edited_by_original = {e["original_sha256"]: f for f, e in edits.items()}
    bad, exact, edited, outside = [], 0, 0, 0
    for record, key, want in pins():
        base = str(pathlib.Path(record).parent)
        path = next((c for c in (key, f"{base}/{key}") if c in files), None)
        if path is None:          # keyed by name, not path: match by content
            path = by_sha.get(want) or edited_by_original.get(want)
        if path is None:
            outside += 1
            continue
        if current[path] == want:
            exact += 1
        elif path in edits and edits[path]["original_sha256"] == want:
            edited += 1
        else:
            bad.append(f"{record} pins {key}: {path} matches neither the pin nor PUBLIC_EDITS.json")
    print(f"freeze records: {exact} pins match byte for byte, {edited} pins are files edited for publication, "
          f"{outside} pins name files outside this release (not checked)")
    py = 0
    for f, e in sorted(edits.items()):
        if f not in files:
            bad.append(f"{f}: listed in PUBLIC_EDITS.json but missing")
            continue
        if current[f] != e["released_sha256"]:
            bad.append(f"{f}: released sha256 differs from PUBLIC_EDITS.json")
        if f.endswith(".py"):
            py += 1
            if syntax_hash(files[f].read_text(), e.get("public_docstrings_in_syntax_hash", False)) != e["syntax_sha256"]:
                bad.append(f"{f}: syntax hash differs from the original's")
    print(f"edited for publication: {len(edits)} files ({sum(e['in_freeze_record'] for e in edits.values())} of them pinned by a freeze record), "
          f"{py} Python files with syntax trees equal to the originals'")
    for b in bad:
        print("MISMATCH", b)
    print("ok" if not bad else f"{len(bad)} mismatches")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""Salted, opaque control ids: 'c' + 8 base32 chars of HMAC(episode salt, logical key)."""
from __future__ import annotations

import base64
import hashlib
import hmac

from ap_sim.app import catalog


def logical_keys() -> list[str]:
    # Every page of every schema; each id is an HMAC of its own key, so adding keys leaves existing ids unchanged
    # (a salt regeneration on a collision would not: checked against every recorded run by the replay tests).
    return [f"act:{a[0]}" for a in catalog.ACTIONS] + [f"nav:{p}" for p, _, _ in catalog.PAGES_V4]


def control_ids(seed: int) -> dict[str, str]:
    """Logical key -> id for one episode; regenerates the salt deterministically on a collision."""
    attempt = 0
    while True:
        salt = hashlib.sha256(f"{seed}:control-salt:{attempt}".encode()).digest()
        ids = {k: "c" + base64.b32encode(hmac.new(salt, k.encode(), hashlib.sha256).digest()).decode()[:8]
               for k in logical_keys()}
        if len(set(ids.values())) == len(ids):
            return ids
        attempt += 1

"""Process groups and GPU cleanup at the operating-system level (added on 2026-11-02).

A vLLM engine is a grandchild of a step. It outlives a parent killed on a timeout, keeps its memory, and holds the log's
pipe open: in the GRPO probe (grpo_probe.py) one held 69.7 GB at 0% utilization until it was killed by hand. So:

    group   Runs a command in its own process group and records the group ID. On timeout, on a signal to this wrapper, or
            when the command exits, whatever is left of the group is ended: SIGINT first (vLLM shuts down gracefully on it
            while it is still responsive), then SIGTERM, about 30 seconds, then SIGKILL. Then as `free`.
    free    nvidia-smi's compute processes, and any vLLM engine, are killed; waits and checks again until the GPU's memory
            is back to near zero. Not freed within about 2 minutes: exit 75, and the session stops.

Every process signalled is logged with its PID and GPU memory, on stdout and in --record (the session's record).
A machine without nvidia-smi has no GPU to free; the group is still ended.

    python scripts/gpu/gpu_cleanup.py free --record FILE
    python scripts/gpu/gpu_cleanup.py group --record FILE [--timeout-minutes M] -- COMMAND...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

FREE_MIB = 512                   # the driver's own use is a few MiB; a process's context alone is more than this
ENGINE_TITLE = "VLLM::"          # vLLM sets this process title on its engine
NOT_FREED = 75                   # the exit code the session stops on
GRACEFUL_SECONDS, TERM_SECONDS, FREE_SECONDS = 15, 30, 120


class Record:
    """One JSON line per event in the session's record, and the same on stdout, which the session's log holds."""

    def __init__(self, path: str | None = None):
        self.path = path

    def __call__(self, event: str, **fields) -> None:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
        line = json.dumps(entry)
        print("gpu cleanup: " + line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def _smi(query: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def memory_used_mib() -> int:
    return max(int(v) for v in _smi("gpu=memory.used"))


def compute_apps() -> dict[int, int]:
    """{pid: MiB} from nvidia-smi --query-compute-apps=pid,used_memory. A memory the driver will not report counts as 0."""
    apps = {}
    for line in _smi("compute-apps=pid,used_memory"):
        pid, _, mib = line.partition(",")
        if pid.strip().isdigit():
            apps[int(pid)] = int(mib.strip()) if mib.strip().isdigit() else 0
    return apps


def _table() -> list[tuple[int, int, int, str, str]]:
    """(pid, ppid, pgid, stat, args) of every process."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,stat=,args="], capture_output=True, text=True, timeout=30, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4 and all(p.isdigit() for p in parts[:3]):
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3], parts[4] if len(parts) > 4 else ""))
    return rows


def group_members(pgid: int) -> set[int]:
    """The group's living processes; a zombie is gone, though nobody may reap it in a container."""
    return {pid for pid, _, group, stat, _ in _table() if group == pgid and not stat.startswith("Z")}


def leftovers(table: list[tuple], gpu_pids: set[int], me: set[int]) -> set[int]:
    """Whatever the driver lists on the GPU and any vLLM engine, with their descendants; never this process or its parents."""
    living = [row for row in table if not row[3].startswith("Z")]
    found = set(gpu_pids) | {row[0] for row in living if row[4].startswith(ENGINE_TITLE)}
    grew = True
    while grew:
        children = {row[0] for row in living if row[1] in found} - found
        found |= children
        grew = bool(children)
    return found - me


def _gpu_memory_of() -> dict[int, int]:
    return compute_apps() if shutil.which("nvidia-smi") else {}


def _args_of(pids: set[int]) -> dict[int, str]:
    return {row[0]: row[4][:80] for row in _table() if row[0] in pids}


def _log_kills(record: Record, sig: signal.Signals, pids: set[int], **fields) -> None:
    memory, args = _gpu_memory_of(), _args_of(pids)
    for pid in sorted(pids):
        record("kill", signal=sig.name, pid=pid, memory_mib=memory.get(pid, 0), command=args.get(pid, ""), **fields)


def stop_group(pgid: int, record: Record, leader: subprocess.Popen | None = None, graceful: float = GRACEFUL_SECONDS,
               term: float = TERM_SECONDS, poll: float = 0.2) -> bool:
    """SIGINT, then SIGTERM, then SIGKILL, to whatever is left of the group. True if the group is gone."""
    for sig, grace in ((signal.SIGINT, graceful), (signal.SIGTERM, term), (signal.SIGKILL, 5)):
        if leader is not None:
            leader.poll()
        members = group_members(pgid)
        if not members:
            return True
        _log_kills(record, sig, members, pgid=pgid)
        try:
            os.killpg(pgid, sig)
        except OSError:
            pass
        end = time.monotonic() + grace
        while time.monotonic() < end:
            if leader is not None:
                leader.poll()
            if not group_members(pgid):
                return True
            time.sleep(poll)
    return not group_members(pgid)


class _Signalled(Exception):
    pass


def run_group(cmd: list[str], record: Record, timeout_seconds: float | None = None, graceful: float = GRACEFUL_SECONDS,
              term: float = TERM_SECONDS) -> int | str:
    """The command's exit code, or "timeout"; its process group is gone when this returns, unless the record says otherwise."""
    def on_signal(signum, _frame):
        raise _Signalled(signum)
    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = proc.pid
    record("group", pgid=pgid, command=" ".join(cmd)[:160])
    code: int | str
    try:
        code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        code = "timeout"
        record("timeout", pgid=pgid, seconds=timeout_seconds)
    except _Signalled as exc:
        code = 128 + exc.args[0]
        record("signalled", pgid=pgid, signal=exc.args[0])
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        if not stop_group(pgid, record, proc, graceful, term):
            record("group-not-gone", pgid=pgid, left=sorted(group_members(pgid)))
    return code


def free_gpu(record: Record, wait_seconds: float = FREE_SECONDS, term: float = TERM_SECONDS, poll: float = 2) -> dict:
    """Kills what is on the GPU until its memory is near zero, or the wait runs out. {"gpu", "free", "memory_mib", "killed"}"""
    if not shutil.which("nvidia-smi"):
        return {"gpu": False, "free": True, "memory_mib": None, "killed": []}
    me, killed = {os.getpid(), os.getppid()}, set()
    end = time.monotonic() + wait_seconds
    while True:
        apps, used = compute_apps(), memory_used_mib()
        stray = leftovers(_table(), set(apps), me)
        if not stray and used < FREE_MIB:
            return {"gpu": True, "free": True, "memory_mib": used, "killed": sorted(killed)}
        if time.monotonic() >= end:
            record("not-freed", memory_mib=used, left=sorted(stray))
            return {"gpu": True, "free": False, "memory_mib": used, "killed": sorted(killed)}
        new = stray - killed
        if new:
            _log_kills(record, signal.SIGTERM, new)
            _signal(new, signal.SIGTERM)
            killed |= new
            limit = min(time.monotonic() + term, end)
            while time.monotonic() < limit and group_alive(new):
                time.sleep(min(poll, 0.2))
        hard = group_alive(stray & killed)                      # what has already had its SIGTERM goes on to SIGKILL
        if hard:
            _log_kills(record, signal.SIGKILL, hard)
            _signal(hard, signal.SIGKILL)
        time.sleep(poll)


def group_alive(pids: set[int]) -> set[int]:
    living = {row[0] for row in _table() if not row[3].startswith("Z")}
    return pids & living


def _signal(pids: set[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("free", "group"))
    ap.add_argument("--record")
    ap.add_argument("--timeout-minutes", type=float, default=0, help="0: none")
    ap.add_argument("command", nargs="*")
    a = ap.parse_args()
    record = Record(a.record)
    code: int | str = 0
    if a.action == "group":
        if not a.command:
            ap.error("group needs a command")
        code = run_group(a.command, record, a.timeout_minutes * 60 or None)
    res = free_gpu(record, float(os.environ.get("GPU_CLEANUP_WAIT_SECONDS", FREE_SECONDS)))
    if not res["free"]:
        print(f"gpu cleanup: STOP: {res['memory_mib']} MiB still in use after the wait (limit {FREE_MIB})", flush=True)
        return NOT_FREED
    return 124 if code == "timeout" else int(code)


if __name__ == "__main__":
    sys.exit(main())

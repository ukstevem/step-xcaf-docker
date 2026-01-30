#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/repo/ui_runs")).resolve()

PLAN_NAME = os.getenv("EXPLODE_PLAN_REL", "explosion_plan.json")
STATE_NAME = os.getenv("EXPLODE_DAEMON_STATE_FILENAME", "_explode_state.json")
LOCK_NAME  = os.getenv("EXPLODE_DAEMON_LOCK_FILENAME", "_explode.lock")

POLL_SEC = float(os.getenv("EXPLODE_DAEMON_POLL_SEC", "2.0"))
MAX_RUNS_PER_TICK = int(os.getenv("EXPLODE_DAEMON_MAX_RUNS_PER_TICK", "20"))
MAX_DIR_SCAN = int(os.getenv("EXPLODE_DAEMON_MAX_DIR_SCAN", "20000"))

# Bounded subprocess runtime (seconds). 0 => no timeout.
EXPLODE_TIMEOUT_SEC = int(os.getenv("EXPLODE_TIMEOUT_SEC", "0"))
PATCH_TIMEOUT_SEC = int(os.getenv("PATCH_TIMEOUT_SEC", "0"))

LOG_NAME = os.getenv("EXPLODE_DAEMON_LOG", "_explode_daemon.log")


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{ts} {msg}\n"
    print("[daemon] " + msg, flush=True)
    try:
        (RUNS_DIR / LOG_NAME).open("a", encoding="utf-8").write(line)
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _acquire_lock(run_dir: Path) -> bool:
    lock = run_dir / LOCK_NAME
    try:
        # O_EXCL ensures only one process can create it
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_lock(run_dir: Path) -> None:
    lock = run_dir / LOCK_NAME
    try:
        lock.unlink(missing_ok=True)
    except Exception:
        pass


def _load_state(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / STATE_NAME
    if not p.is_file():
        return {}
    try:
        return _read_json(p)
    except Exception:
        return {}


def _save_state(run_dir: Path, st: Dict[str, Any]) -> None:
    _write_json_atomic(run_dir / STATE_NAME, st)


def _should_process(run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    plan = run_dir / PLAN_NAME
    if not plan.is_file():
        return (False, None, None)

    # Ensure plan is valid JSON before hashing (avoids racing half-written file)
    try:
        data = _read_json(plan)
        if not isinstance(data, dict):
            return (False, None, None)
    except Exception:
        return (False, None, None)

    # Now it is safe to hash
    plan_hash = _sha256_file(plan)

    st = _load_state(run_dir)
    last_hash = str(st.get("last_plan_hash") or "").strip()
    fail_hash = str(st.get("last_fail_hash") or "").strip()

    # If we already tried (and failed) this exact hash, don't keep retrying
    if plan_hash and plan_hash == fail_hash:
        return (False, plan_hash, last_hash or None)

    # If plan differs from last success, process it
    if plan_hash and plan_hash != last_hash:
        return (True, plan_hash, last_hash or None)

    return (False, plan_hash, last_hash or None)



def _run_cmd(cmd: list[str], timeout_sec: int) -> int:
    try:
        p = subprocess.run(cmd, timeout=(timeout_sec if timeout_sec > 0 else None))
        return int(p.returncode)
    except subprocess.TimeoutExpired:
        return 124
    except Exception:
        return 125


def _process_run(run_id: str, run_dir: Path, plan_hash: str) -> None:
    if not _acquire_lock(run_dir):
        _log(f"skip {run_id} (locked)")
        return

    try:
        _log(f"start {run_id} plan_hash={plan_hash[:12]}…")

        rc1 = _run_cmd(["python", "/repo/explode_multibody.py", "--run_id", run_id], EXPLODE_TIMEOUT_SEC)
        rc2 = _run_cmd(["python", "/repo/patch_bom_tree_for_exploded.py", "--run_id", run_id], PATCH_TIMEOUT_SEC)

        st = _load_state(run_dir)
        st["last_attempt_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["last_explode_rc"] = rc1
        st["last_patch_rc"] = rc2

        if rc1 == 0 and rc2 == 0:
            st["last_plan_hash"] = plan_hash
            st["last_success_utc"] = st["last_attempt_utc"]
            if "last_fail_hash" in st:
                del st["last_fail_hash"]
            _save_state(run_dir, st)
            _log(f"done  {run_id} OK")
        else:
            # latch this hash so we don't hammer the same bad plan every poll tick
            st["last_fail_hash"] = plan_hash
            _save_state(run_dir, st)
            _log(f"done  {run_id} FAIL rc_explode={rc1} rc_patch={rc2}")


    finally:
        _release_lock(run_dir)


def main() -> int:
    _log(f"daemon up RUNS_DIR={RUNS_DIR} plan={PLAN_NAME} poll={POLL_SEC}s")
    while True:
        try:
            if not RUNS_DIR.is_dir():
                _log("RUNS_DIR missing; sleeping")
                time.sleep(max(POLL_SEC, 1.0))
                continue

            # bounded directory scan
            run_dirs = []
            n = 0
            for p in RUNS_DIR.iterdir():
                if p.is_dir():
                    run_dirs.append(p)
                    n += 1
                    if n >= MAX_DIR_SCAN:
                        break

            run_dirs.sort(key=lambda d: d.name)

            processed = 0
            for rd in run_dirs:
                if processed >= MAX_RUNS_PER_TICK:
                    break

                run_id = rd.name
                ok, plan_hash, _last = _should_process(rd)
                if not ok or not plan_hash:
                    continue

                _process_run(run_id, rd, plan_hash)
                processed += 1

            time.sleep(POLL_SEC)

        except Exception as e:
            _log(f"tick error: {type(e).__name__}: {e}")
            time.sleep(max(POLL_SEC, 1.0))

    # unreachable
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())

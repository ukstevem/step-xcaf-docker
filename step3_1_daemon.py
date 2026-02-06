#!/usr/bin/env python3
"""
step3_1_daemon.py  (Wrapper daemon for Step 3.1)

Purpose
- Runs Step 3.1 (parts_index.json builder) as a daemon over RUNS_DIR.
- This avoids changing your existing one-shot tool:
    python step3_1_worker.py <run_dir>

Inputs (per run_dir)
- Whatever step3_1_worker.py expects inside the run folder (e.g. assets/stl manifests)

Outputs (per run_dir)
- parts_index.json (produced by step3_1_worker.py)

Notes
- Deterministic, bounded scanning
- Lock per run to avoid concurrent rebuild
- Optional: only rebuild when inputs change (hash latch)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/app/ui_runs")).resolve()
POLL_SEC = float(os.getenv("POLL_SECS", "2.0"))

# bounded scanning
MAX_DIR_SCAN = int(os.getenv("STEP3_1_DAEMON_MAX_DIR_SCAN", "20000"))
MAX_RUNS_PER_TICK = int(os.getenv("STEP3_1_DAEMON_MAX_RUNS_PER_TICK", "40"))

# output
OUT_REL = os.getenv("PARTS_INDEX_REL", "parts_index.json")

# daemon state/locks (per run)
STATE_NAME = os.getenv("STEP3_1_DAEMON_STATE_FILENAME", "_step3_1_state.json")
LOCK_NAME = os.getenv("STEP3_1_DAEMON_LOCK_FILENAME", "_step3_1.lock")
LOG_NAME = os.getenv("STEP3_1_DAEMON_LOG", "_step3_1_daemon.log")

# optional: only attempt if these exist (set in env if you want)
REQ_FILES = [s.strip() for s in os.getenv("STEP3_1_REQUIRED_FILES", "").split(",") if s.strip()]

# command to run existing one-shot tool
STEP3_1_CMD = os.getenv("STEP3_1_CMD", "python /repo/step3_1_worker.py").strip()


def _utc_iso_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    ts = _utc_iso_z()
    line = f"{ts} {msg}\n"
    print(f"[step3_1_daemon] {msg}", flush=True)
    try:
        (RUNS_DIR / LOG_NAME).open("a", encoding="utf-8").write(line)
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


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
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_lock(run_dir: Path) -> None:
    try:
        (run_dir / LOCK_NAME).unlink(missing_ok=True)
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


def _run_dir_ready(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    if REQ_FILES:
        for rel in REQ_FILES:
            p = run_dir / rel
            if not p.is_file():
                return False
            try:
                if p.stat().st_size <= 0:
                    return False
            except Exception:
                return False
    return True


def _inputs_hash(run_dir: Path) -> Optional[str]:
    """
    If you set STEP3_1_REQUIRED_FILES, we hash those to avoid re-running on every tick.
    If you don't set it, we fall back to: "run if output missing".
    """
    if not REQ_FILES:
        return None
    try:
        h = hashlib.sha256()
        for rel in REQ_FILES:
            p = run_dir / rel
            h.update(_sha256_file(p).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return None


def _should_process(run_dir: Path) -> Tuple[bool, Optional[str]]:
    outp = run_dir / OUT_REL

    if not _run_dir_ready(run_dir):
        return (False, None)

    ih = _inputs_hash(run_dir)

    # If we don't know inputs hash, only build when output missing
    if ih is None:
        return (not outp.is_file(), None)

    st = _load_state(run_dir)
    last = str(st.get("last_inputs_hash") or "")
    fail = str(st.get("last_fail_hash") or "")

    if ih and ih == fail:
        return (False, ih)

    if not outp.is_file():
        return (True, ih)

    if ih and ih != last:
        return (True, ih)

    return (False, ih)


def _invoke_step3_1(run_dir: Path) -> None:
    # STEP3_1_CMD may include args; we append run_dir
    cmd = STEP3_1_CMD.split() + [str(run_dir)]
    res = subprocess.run(
        cmd,
        cwd="/repo",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=None,
    )
    if res.returncode != 0:
        raise RuntimeError(f"step3_1_worker failed rc={res.returncode}: {res.stdout[-2000:]}")


def _process_run(run_id: str, run_dir: Path) -> bool:
    if not _acquire_lock(run_dir):
        return False

    try:
        ok, ih = _should_process(run_dir)
        if not ok:
            return False

        _log(f"start {run_id}…")
        _invoke_step3_1(run_dir)

        st = _load_state(run_dir)
        st["last_attempt_utc"] = _utc_iso_z()
        if ih:
            st["last_inputs_hash"] = ih
        st["last_success_utc"] = st["last_attempt_utc"]
        if "last_fail_hash" in st:
            del st["last_fail_hash"]
        if "last_error" in st:
            del st["last_error"]
        _save_state(run_dir, st)

        _log(f"done  {run_id} OK -> {OUT_REL}")
        return True

    except Exception as e:
        st = _load_state(run_dir)
        st["last_attempt_utc"] = _utc_iso_z()
        if ih:
            st["last_fail_hash"] = ih
        st["last_error"] = f"{type(e).__name__}: {e}"
        _save_state(run_dir, st)
        _log(f"done  {run_id} FAIL {type(e).__name__}: {e}")
        return False

    finally:
        _release_lock(run_dir)


def main() -> int:
    _log(f"daemon up RUNS_DIR={RUNS_DIR} poll={POLL_SEC}s out={OUT_REL}")
    while True:
        try:
            if not RUNS_DIR.is_dir():
                _log("RUNS_DIR missing; sleeping")
                time.sleep(max(POLL_SEC, 1.0))
                continue

            run_dirs: List[Path] = []
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
                if _process_run(rd.name, rd):
                    processed += 1

            time.sleep(POLL_SEC)

        except Exception as e:
            _log(f"tick error: {type(e).__name__}: {e}")
            time.sleep(max(POLL_SEC, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())

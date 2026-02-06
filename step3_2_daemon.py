#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/app/ui_runs")).resolve()
POLL_SEC = float(os.getenv("POLL_SECS", "2.0"))

MAX_DIR_SCAN = int(os.getenv("STEP3_2_DAEMON_MAX_DIR_SCAN", "20000"))
MAX_RUNS_PER_TICK = int(os.getenv("STEP3_2_DAEMON_MAX_RUNS_PER_TICK", "40"))

BOM_GLOBAL_REL = os.getenv("BOM_GLOBAL_REL", "bom_global_exploded.json")
PARTS_INDEX_REL = os.getenv("PARTS_INDEX_REL", "parts_index.json")
CATEGORIES_REL = os.getenv("CATEGORIES_REL", "categories.json")

STATE_NAME = os.getenv("STEP3_2_DAEMON_STATE_FILENAME", "_step3_2_state.json")
LOCK_NAME = os.getenv("STEP3_2_DAEMON_LOCK_FILENAME", "_step3_2.lock")
LOG_NAME = os.getenv("STEP3_2_DAEMON_LOG", "_step3_2_daemon.log")

STEP3_2_CMD = os.getenv("STEP3_2_CMD", "python /repo/categoriser_worker.py").strip()


def _utc_iso_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    line = f"{_utc_iso_z()} {msg}\n"
    print(f"[step3_2_daemon] {msg}", flush=True)
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


def _required_ready(run_dir: Path) -> bool:
    p_bom = run_dir / BOM_GLOBAL_REL
    p_pi = run_dir / PARTS_INDEX_REL
    if not (p_bom.is_file() and p_pi.is_file()):
        return False
    try:
        if p_bom.stat().st_size <= 0 or p_pi.stat().st_size <= 0:
            return False
    except Exception:
        return False
    return True


def _inputs_hash(run_dir: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        h.update(_sha256_file(run_dir / BOM_GLOBAL_REL).encode("utf-8"))
        h.update(_sha256_file(run_dir / PARTS_INDEX_REL).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return None


def _should_process(run_dir: Path) -> Tuple[bool, Optional[str]]:
    if not _required_ready(run_dir):
        return (False, None)

    ih = _inputs_hash(run_dir)
    if not ih:
        return (False, None)

    outp = run_dir / CATEGORIES_REL
    st = _load_state(run_dir)
    last = str(st.get("last_inputs_hash") or "")
    fail = str(st.get("last_fail_hash") or "")

    if ih == fail:
        return (False, ih)

    if not outp.is_file():
        return (True, ih)

    if ih != last:
        return (True, ih)

    return (False, ih)


def _invoke_categoriser(run_dir: Path) -> None:
    cmd = STEP3_2_CMD.split() + [str(run_dir)]
    res = subprocess.run(
        cmd,
        cwd="/repo",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=None,
    )
    if res.returncode != 0:
        tail = (res.stdout or "")[-2000:]
        raise RuntimeError(f"categoriser failed rc={res.returncode}: {tail}")


def _process_run(run_id: str, run_dir: Path) -> bool:
    if not _acquire_lock(run_dir):
        return False

    ih: Optional[str] = None
    try:
        ok, ih = _should_process(run_dir)
        if not ok:
            return False

        _log(f"start {run_id}…")
        _invoke_categoriser(run_dir)

        st = _load_state(run_dir)
        st["last_attempt_utc"] = _utc_iso_z()
        st["last_inputs_hash"] = ih
        st["last_success_utc"] = st["last_attempt_utc"]
        st.pop("last_fail_hash", None)
        st.pop("last_error", None)
        _save_state(run_dir, st)

        _log(f"done  {run_id} OK -> {CATEGORIES_REL}")
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
    _log(f"daemon up RUNS_DIR={RUNS_DIR} poll={POLL_SEC}s out={CATEGORIES_REL}")
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

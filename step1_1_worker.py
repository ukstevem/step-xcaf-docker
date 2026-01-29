#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _utc_hms() -> str:
    # progress.log is “human readable”; keep it simple like your other lines
    return time.strftime("%H:%M:%S", time.gmtime())


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append_progress(run_dir: Path, msg: str) -> None:
    prog = run_dir / os.environ.get("PROGRESS_REL", "progress.log")
    ts = time.strftime("%H:%M:%S", time.localtime())
    with prog.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg.rstrip()}\n")
        f.flush()


def _env_csv(name: str, default: str) -> Tuple[str, ...]:
    raw = (os.environ.get(name) or default).strip()
    if not raw:
        return tuple()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _try_lock(run_dir: Path, lock_name: str) -> bool:
    lock = run_dir / lock_name
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _unlock(run_dir: Path, lock_name: str) -> None:
    try:
        (run_dir / lock_name).unlink(missing_ok=True)
    except Exception:
        pass


@dataclass(frozen=True)
class Cfg:
    runs_dir: Path
    status_rel: str
    sleep_sec: float
    trigger_stages: Tuple[str, ...]
    heartbeat_sec: float
    export_cmd_tpl: str


def _load_cfg() -> Cfg:
    runs_dir = Path(os.environ.get("RUNS_DIR", "/app/ui_runs"))
    status_rel = os.environ.get("STATUS_REL", "status.json")
    sleep_sec = float(os.environ.get("WORKER_SLEEP_SEC", "2.0"))
    trigger_stages = _env_csv("STL_EXPORT_TRIGGER_STAGES", "ready")
    heartbeat_sec = float(os.environ.get("STL_EXPORT_HEARTBEAT_SEC", "10.0"))
    export_cmd_tpl = os.environ.get(
        "EXPORT_STL_CMD",
        'python /repo/export_stl_xcaf.py --step-path "{step}" --out-dir "{out}" --xcaf-json "{xcaf}"',
    )
    return Cfg(runs_dir, status_rel, sleep_sec, trigger_stages, heartbeat_sec, export_cmd_tpl)


def _is_ready_for_export(run_dir: Path, cfg: Cfg) -> bool:
    st = _read_json(run_dir / cfg.status_rel) or {}
    stage = str(st.get("stage") or "")
    if stage not in cfg.trigger_stages:
        return False

    # prerequisites
    if not (run_dir / "input.step").exists():
        return False
    if not (run_dir / "xcaf_instances.json").exists():
        return False

    # already done or already failed? (don’t loop forever)
    marker = _read_json(run_dir / "stl_export_status.json") or {}
    mstage = str(marker.get("stage") or "")
    if mstage in ("done", "failed"):
        return False

    return True

def _refresh_tree(run_dir: Path) -> None:
    # Rebuild occ_tree.json now that assets_manifest.json has stl_path entries.
    cmd = ["python", "/repo/build_occ_tree.py", "--run-id", run_dir.name, "--runs-root", str(run_dir.parent)]
    _append_progress(run_dir, "Tree: refreshing occ_tree.json (post-manifest)…")
    subprocess.run(cmd, check=True)
    _append_progress(run_dir, "Tree: refresh done.")


def _run_export(run_dir: Path, cfg: Cfg) -> None:
    lock_name = ".stl_export_lock"
    if not _try_lock(run_dir, lock_name):
        return

    try:
        marker_path = run_dir / "stl_export_status.json"
        _write_json(marker_path, {"stage": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

        step = run_dir / "input.step"
        xcaf = run_dir / "xcaf_instances.json"
        out = run_dir

        cmd_str = cfg.export_cmd_tpl.format(step=str(step), xcaf=str(xcaf), out=str(out))
        argv = shlex.split(cmd_str)

        _append_progress(run_dir, "STL export: starting…")
        _append_progress(run_dir, f"STL export: CMD: {cmd_str}")

        # run, streaming stdout->progress
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None

        for line in p.stdout:
            line = line.rstrip()
            if line:
                _append_progress(run_dir, f"STL export: {line}")

        rc = p.wait()
        if rc == 0:
            _append_progress(run_dir, "STL export: done.")
            _write_json(marker_path, {"stage": "done", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        else:
            _append_progress(run_dir, f"STL export: FAILED (exit {rc})")
            _write_json(
                marker_path,
                {"stage": "failed", "exit_code": rc, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            )

    finally:
        _unlock(run_dir, lock_name)


def main() -> int:
    cfg = _load_cfg()
    print(f"[step1_1_worker] watching: {cfg.runs_dir}")
    print(f"[step1_1_worker] trigger stages: {cfg.trigger_stages}")

    last_hb = 0.0
    while True:
        now = time.time()
        if now - last_hb >= cfg.heartbeat_sec:
            print(f"[step1_1_worker] heartbeat {time.strftime('%H:%M:%S')}")
            last_hb = now

        try:
            if cfg.runs_dir.exists():
                for run_dir in sorted(cfg.runs_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    if _is_ready_for_export(run_dir, cfg):
                        _run_export(run_dir, cfg)
                        _refresh_tree(run_dir)
        except Exception as e:
            # keep worker alive no matter what
            print(f"[step1_1_worker] warning: {e}")

        time.sleep(cfg.sleep_sec)


if __name__ == "__main__":
    raise SystemExit(main())

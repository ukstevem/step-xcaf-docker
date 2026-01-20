from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List


DECISION_ENUM = ("keep_as_one", "explode", "defer")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_decisions_csv(path: Path, decisions: List[Dict[str, str]]) -> int:
    cols = ["def_sig", "decision", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        n = 0
        for d in decisions:
            sig = (d.get("def_sig") or "").strip()
            if not sig:
                continue
            decision = (d.get("decision") or "defer").strip()
            note = (d.get("note") or "").strip()
            if decision not in DECISION_ENUM:
                decision = "defer"
            w.writerow({"def_sig": sig, "decision": decision, "note": note})
            n += 1
    return n


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


class Handler(SimpleHTTPRequestHandler):
    # These will be set on the class before server starts
    OUTDIR: Path = Path(".")
    UI_DIR: Path = Path(".")
    DECISIONS_CSV: Path = Path(".")
    DECISIONS_LOG: Path = Path(".")

    def _send_json(self, obj: Any, code: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/api/items":
            items_path = self.UI_DIR / "items.json"
            if not items_path.exists():
                return self._send_json({"error": "items.json missing - run step4_multibody_ui_build.py"}, 500)
            try:
                data = json.loads(items_path.read_text(encoding="utf-8"))
                # Serve just items array
                return self._send_json({"items": data.get("items", [])})
            except Exception:
                return self._send_json({"error": "failed to read items.json"}, 500)

        # Serve UI static (index/app/styles) from UI_DIR
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/save":
            return self._send_json({"error": "not found"}, 404)

        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._send_json({"error": "bad json"}, 400)

        decisions = body.get("decisions")
        if not isinstance(decisions, list):
            return self._send_json({"error": "decisions must be list"}, 400)

        # Normalize and write CSV
        norm: List[Dict[str, str]] = []
        log_rows: List[Dict[str, Any]] = []
        ts = _now_utc_iso()

        for d in decisions:
            if not isinstance(d, dict):
                continue
            sig = (d.get("def_sig") or "").strip()
            if not sig:
                continue
            decision = (d.get("decision") or "defer").strip()
            note = (d.get("note") or "").strip()
            if decision not in DECISION_ENUM:
                decision = "defer"

            norm.append({"def_sig": sig, "decision": decision, "note": note})
            log_rows.append(
                {
                    "ts_utc": ts,
                    "def_sig": sig,
                    "decision": decision,
                    "note": note,
                    "source": "ui",
                }
            )

        updated = _write_decisions_csv(self.DECISIONS_CSV, norm)
        _append_jsonl(self.DECISIONS_LOG, log_rows)

        return self._send_json({"ok": True, "updated": updated})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory (e.g. out or /out)")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--bind", default="0.0.0.0")
    ns = ap.parse_args()

    outdir = Path(ns.outdir)
    ui_dir = outdir / "review_ui"

    if not ui_dir.exists():
        raise SystemExit(f"Missing UI dir: {ui_dir} (run build first)")

    # Serve files from ui_dir
    Handler.OUTDIR = outdir
    Handler.UI_DIR = ui_dir
    Handler.DECISIONS_CSV = outdir / "review" / "multibody_decisions.csv"
    Handler.DECISIONS_LOG = outdir / "review" / "multibody_decisions_log.jsonl"

    # Switch CWD so SimpleHTTPRequestHandler serves ui_dir
    import os
    os.chdir(str(ui_dir))

    httpd = HTTPServer((ns.bind, ns.port), Handler)
    print(f"[ok] serving: http://localhost:{ns.port}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

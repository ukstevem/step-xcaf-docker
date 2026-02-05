#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Config (env-driven)
# -----------------------------

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/app/ui_runs")).resolve()

BOM_GLOBAL_REL = os.getenv("BOM_GLOBAL_REL", "bom_global_exploded.json")
PARTS_INDEX_REL = os.getenv("PARTS_INDEX_REL", "parts_index.json")
CATEGORIES_REL = os.getenv("CATEGORIES_REL", "categories.json")
BOM_FLAT_REL = os.getenv("BOM_FLAT_REL", "bom_flat.json")

STATE_NAME = os.getenv("BOM_FLAT_DAEMON_STATE_FILENAME", "_bom_flat_state.json")
LOCK_NAME = os.getenv("BOM_FLAT_DAEMON_LOCK_FILENAME", "_bom_flat.lock")
LOG_NAME = os.getenv("BOM_FLAT_DAEMON_LOG", "_bom_flat_daemon.log")

POLL_SEC = float(os.getenv("POLL_SECS", "2.0"))
MAX_RUNS_PER_TICK = int(os.getenv("BOM_FLAT_DAEMON_MAX_RUNS_PER_TICK", "40"))
MAX_DIR_SCAN = int(os.getenv("BOM_FLAT_DAEMON_MAX_DIR_SCAN", "20000"))

MAX_BOM_ROWS = int(os.getenv("BOM_FLAT_MAX_BOM_ROWS", "300000"))
MAX_OUT_ROWS = int(os.getenv("BOM_FLAT_MAX_OUT_ROWS", "200000"))
MAX_MEMBER_SIGS_PER_ROW = int(os.getenv("BOM_FLAT_MAX_MEMBER_SIGS_PER_ROW", "40"))
MAX_WHERE_USED_PER_ROW = int(os.getenv("BOM_FLAT_MAX_WHERE_USED_PER_ROW", "40"))

MIN_INPUT_AGE_SEC = float(os.getenv("BOM_FLAT_MIN_INPUT_AGE_SEC", "0.25"))

SCHEMA_OUT = "bom_flat.v1"


# -----------------------------
# Logging / IO utils
# -----------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{ts} {msg}\n"
    print("[bom_flat] " + msg, flush=True)
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


def _utc_iso_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _sorted3(vals: Any) -> Optional[List[float]]:
    if not isinstance(vals, list) or len(vals) != 3:
        return None
    try:
        a = [float(vals[0]), float(vals[1]), float(vals[2])]
        a.sort(reverse=True)
        return a
    except Exception:
        return None


def _extract_free_from_common(common_key: str) -> Optional[str]:
    # common:FREE:CHIRAL
    if not isinstance(common_key, str):
        return None
    parts = common_key.split(":")
    if len(parts) != 3:
        return None
    if parts[0] != "common":
        return None
    free = parts[1].strip()
    return free if free else None


# -----------------------------
# Category helpers
# -----------------------------

def _cat_info(categories: Dict[str, Any], sig_key: str) -> Tuple[str, Optional[str], bool, Optional[float]]:
    it = (categories.get("items") or {}).get(sig_key)
    if not isinstance(it, dict):
        return ("unknown", None, False, None)

    eff = it.get("effective") or {}
    auto = it.get("auto") or {}

    cat = str(eff.get("category") or "unknown")
    src = eff.get("source")
    split = bool(auto.get("chirality_split") or False)
    conf = auto.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except Exception:
        conf_f = None

    return (cat, str(src) if src is not None else None, split, conf_f)


def _group_key_for_sig(sig_key: str, parts_index: Dict[str, Any], categories: Dict[str, Any]) -> str:
    p = (parts_index.get("items") or {}).get(sig_key) or {}

    free_sig = p.get("free_sig")
    group_id = p.get("group_id")
    chiral_sig = p.get("chiral_sig")
    common_key = p.get("common_key")

    _cat, _src, chirality_split, _conf = _cat_info(categories, sig_key)

    if chirality_split:
        # Handed: keep chiral split; prefer existing stable common_key
        if isinstance(common_key, str) and common_key:
            return common_key
        if isinstance(free_sig, str) and free_sig and isinstance(chiral_sig, str) and chiral_sig:
            return f"common:{free_sig}:{chiral_sig}"
        if isinstance(group_id, str) and group_id and isinstance(chiral_sig, str) and chiral_sig:
            return f"common:{group_id}:{chiral_sig}"
        return f"sig:{sig_key}"

    # Non-handed: merge by free_sig if possible
    if isinstance(free_sig, str) and free_sig:
        return f"free:{free_sig}"
    if isinstance(group_id, str) and group_id:
        return f"group:{group_id}"
    if isinstance(common_key, str) and common_key:
        free = _extract_free_from_common(common_key)
        if free:
            return f"free:{free}"
        return common_key

    return f"sig:{sig_key}"


def _canonical_rep_sig_for_group(group_key: str, parts_index: Dict[str, Any], member_sigs_sorted: List[str]) -> str:
    groups = parts_index.get("groups") or {}

    gid: Optional[str] = None
    if group_key.startswith("free:"):
        gid = group_key[len("free:") :]
    elif group_key.startswith("group:"):
        gid = group_key[len("group:") :]

    if gid and isinstance(groups.get(gid), dict):
        rep = groups[gid].get("canonical_member_sig_key")
        if isinstance(rep, str) and rep:
            return rep

    return member_sigs_sorted[0]


# -----------------------------
# Core build
# -----------------------------

def _build_bom_flat(bom: Dict[str, Any], parts_index: Dict[str, Any], categories: Dict[str, Any]) -> Dict[str, Any]:
    items = bom.get("items")
    if not isinstance(items, list):
        raise RuntimeError("bom_global_exploded.json: missing/invalid 'items' list")
    if len(items) > MAX_BOM_ROWS:
        raise RuntimeError(f"bom_global_exploded.json: too many rows ({len(items)} > {MAX_BOM_ROWS})")

    # Robust exploded-parent detection:
    # Any row with from_parent_def_sig is a subpart row => its parent must be suppressed.
    exploded_parent_sigs: Dict[str, int] = {}
    for r in items:
        if not isinstance(r, dict):
            continue
        p = r.get("from_parent_def_sig")
        if isinstance(p, str) and p:
            exploded_parent_sigs[p] = exploded_parent_sigs.get(p, 0) + 1

    skipped_empty = 0
    skipped_exploded_parents = 0
    flattened: List[Dict[str, Any]] = []

    for r in items:
        if not isinstance(r, dict):
            continue

        sig = r.get("ref_def_sig")
        if not isinstance(sig, str) or not sig:
            skipped_empty += 1
            continue

        if r.get("shape_kind") == "EMPTY":
            skipped_empty += 1
            continue

        is_subpart = isinstance(r.get("from_parent_def_sig"), str) and bool(r.get("from_parent_def_sig"))
        if (not is_subpart) and (sig in exploded_parent_sigs):
            skipped_exploded_parents += 1
            continue

        flattened.append(r)

    # Bucket
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in flattened:
        sig = str(r["ref_def_sig"])
        gk = _group_key_for_sig(sig, parts_index, categories)

        b = buckets.get(gk)
        if b is None:
            b = {"qty": 0, "member_sigs": set(), "example_names": set(), "sample_rows": []}
            buckets[gk] = b

        b["qty"] += _safe_int(r.get("qty_total"), 0)
        b["member_sigs"].add(sig)

        dn = r.get("def_name")
        if isinstance(dn, str) and dn:
            b["example_names"].add(dn)

        if len(b["sample_rows"]) < 3:
            b["sample_rows"].append(r)

    if len(buckets) > MAX_OUT_ROWS:
        raise RuntimeError(f"bom_flat: too many output groups ({len(buckets)} > {MAX_OUT_ROWS})")

    pitems = parts_index.get("items") or {}

    out_rows: List[Dict[str, Any]] = []
    for gk in sorted(buckets.keys()):
        b = buckets[gk]
        member_sigs_sorted = sorted(list(b["member_sigs"]))
        rep_sig = _canonical_rep_sig_for_group(gk, parts_index, member_sigs_sorted)

        # Representative STL + bbox
        p = pitems.get(rep_sig) or {}

        rep_stl = p.get("rep_stl_url")
        if not isinstance(rep_stl, str) or not rep_stl:
            rep_stl = None
            for sr in b["sample_rows"]:
                u = sr.get("stl_url")
                if isinstance(u, str) and u:
                    rep_stl = u
                    break

        bbox_sorted = p.get("bbox_sorted")
        if not (isinstance(bbox_sorted, list) and len(bbox_sorted) == 3):
            bbox_sorted = None
            for sr in b["sample_rows"]:
                bb = sr.get("bbox_mm") or {}
                s3 = _sorted3(bb.get("size"))
                if s3:
                    bbox_sorted = s3
                    break

        # Category from representative
        cat, src, split, conf = _cat_info(categories, rep_sig)

        example_name = ""
        if b["example_names"]:
            example_name = sorted([x for x in b["example_names"] if x])[0]

        # Optional where_used summary from rep
        where_used = p.get("where_used")
        where_used_out: Optional[List[Dict[str, Any]]] = None
        if isinstance(where_used, list) and where_used:
            where_used_out = []
            for wu in where_used[:MAX_WHERE_USED_PER_ROW]:
                if isinstance(wu, dict):
                    where_used_out.append(
                        {
                            "def_name": wu.get("def_name"),
                            "from_parent_def_name": wu.get("from_parent_def_name"),
                            "from_parent_def_sig": wu.get("from_parent_def_sig"),
                            "occ_label_sample": wu.get("occ_label_sample"),
                        }
                    )

        row: Dict[str, Any] = {
            "common_part_id": gk,
            "qty": int(b["qty"]),
            "category": cat,
            "category_source": src,
            "category_confidence": conf,
            "chirality_split": bool(split),
            "bbox_sorted_mm": bbox_sorted,
            "representative_sig_key": rep_sig,
            "representative_stl_url": rep_stl,
            "example_def_name": example_name,
            "members_count": len(member_sigs_sorted),
        }

        if len(member_sigs_sorted) <= MAX_MEMBER_SIGS_PER_ROW:
            row["member_sig_keys"] = member_sigs_sorted
        else:
            row["member_sig_keys"] = member_sigs_sorted[:MAX_MEMBER_SIGS_PER_ROW]
            row["member_sig_keys_truncated"] = True

        if where_used_out is not None:
            row["where_used"] = where_used_out
            row["where_used_truncated"] = (len(where_used) > MAX_WHERE_USED_PER_ROW)

        out_rows.append(row)

    out_rows.sort(key=lambda r: (str(r.get("category") or ""), str(r.get("common_part_id") or ""), str(r.get("representative_sig_key") or "")))

    return {
        "schema": SCHEMA_OUT,
        "generated_at": _utc_iso_z(),
        "sources": {
            "bom_global_exploded": BOM_GLOBAL_REL,
            "parts_index": PARTS_INDEX_REL,
            "categories": CATEGORIES_REL,
        },
        "summary": {
            "bom_rows_in": len(items),
            "exploded_parent_defs": len(exploded_parent_sigs),
            "flattened_rows": len(flattened),
            "skipped_empty_or_missing_sig": skipped_empty,
            "skipped_exploded_parent_rows": skipped_exploded_parents,
            "flat_groups_out": len(out_rows),
        },
        "items": out_rows,
    }


# -----------------------------
# Trigger logic (state hash)
# -----------------------------

def _should_process(run_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    p_bom = run_dir / BOM_GLOBAL_REL
    p_pi  = run_dir / PARTS_INDEX_REL
    p_cat = run_dir / CATEGORIES_REL
    p_out = run_dir / BOM_FLAT_REL

    # Required inputs must exist and be non-empty
    if not (p_bom.is_file() and p_pi.is_file() and p_cat.is_file()):
        return (False, None, None)

    try:
        if p_bom.stat().st_size <= 0 or p_pi.stat().st_size <= 0 or p_cat.stat().st_size <= 0:
            return (False, None, None)
    except Exception:
        return (False, None, None)

    # Hash inputs. If any file is mid-write/unreadable, hash may fail -> skip this tick.
    try:
        h = hashlib.sha256()
        h.update(_sha256_file(p_bom).encode("utf-8"))
        h.update(_sha256_file(p_pi).encode("utf-8"))
        h.update(_sha256_file(p_cat).encode("utf-8"))
        combo_hash = h.hexdigest()
    except Exception:
        return (False, None, None)

    st = _load_state(run_dir)
    last_hash = str(st.get("last_inputs_hash") or "").strip()
    fail_hash = str(st.get("last_fail_hash") or "").strip()

    # If output is missing, rebuild (unless this hash is latched as failing)
    if not p_out.is_file():
        if combo_hash and combo_hash == fail_hash:
            return (False, combo_hash, last_hash or None)
        return (True, combo_hash, last_hash or None)

    # Normal behaviour: avoid hammering known-bad inputs
    if combo_hash and combo_hash == fail_hash:
        return (False, combo_hash, last_hash or None)

    # Rebuild only when inputs change
    if combo_hash and combo_hash != last_hash:
        return (True, combo_hash, last_hash or None)

    return (False, combo_hash, last_hash or None)




def _process_run_if_needed(run_id: str, run_dir: Path) -> bool:
    """
    Returns True if we actually processed the run (wrote output), else False.
    Lock is acquired first to avoid races between multiple daemons.
    """
    if not _acquire_lock(run_dir):
        _log(f"skip {run_id} (locked)")
        return False

    try:
        ok, inputs_hash, _last = _should_process(run_dir)
        if not ok or not inputs_hash:
            return False

        _log(f"start {run_id} inputs_hash={inputs_hash[:12]}…")

        p_bom = run_dir / BOM_GLOBAL_REL
        p_pi = run_dir / PARTS_INDEX_REL
        p_cat = run_dir / CATEGORIES_REL
        p_out = run_dir / BOM_FLAT_REL

        bom = _read_json(p_bom)
        parts_index = _read_json(p_pi)
        categories = _read_json(p_cat)

        out = _build_bom_flat(bom=bom, parts_index=parts_index, categories=categories)
        _write_json_atomic(p_out, out)

        st = _load_state(run_dir)
        st["last_attempt_utc"] = _utc_iso_z()
        st["last_inputs_hash"] = inputs_hash
        st["last_success_utc"] = st["last_attempt_utc"]
        if "last_fail_hash" in st:
            del st["last_fail_hash"]
        if "last_error" in st:
            del st["last_error"]
        _save_state(run_dir, st)

        _log(f"done  {run_id} OK -> {BOM_FLAT_REL} groups={len(out.get('items') or [])}")
        return True

    except Exception as e:
        # Only latch fail hash if we had computed one
        try:
            ok2, inputs_hash2, _ = _should_process(run_dir)
            ih = inputs_hash2 or ""
        except Exception:
            ih = ""
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
    _log(f"daemon up RUNS_DIR={RUNS_DIR} poll={POLL_SEC}s out={BOM_FLAT_REL}")
    while True:
        try:
            if not RUNS_DIR.is_dir():
                _log("RUNS_DIR missing; sleeping")
                time.sleep(max(POLL_SEC, 1.0))
                continue

            # bounded directory scan
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

                run_id = rd.name

                did = _process_run_if_needed(run_id, rd)
                if did:
                    processed += 1

            time.sleep(POLL_SEC)

        except Exception as e:
            _log(f"tick error: {type(e).__name__}: {e}")
            time.sleep(max(POLL_SEC, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())

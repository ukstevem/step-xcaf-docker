#!/usr/bin/env python3
"""
Step 4 (additive): procurement CSVs from:
  - /out/xcaf_instances.json
  - /out/stl_manifest.json

Outputs:
  /out/step4_parts.csv
  /out/step4_proc_{plates|sections|hardware|handrail|stairtreads|floor_grating|unknown}.csv

Classification:
  1) Name-first keywords
  2) If still unknown: geometry + massprops fallback
     - sections: long slender (L / mid >= SECTION_LEN_RATIO)
     - plates: plate-like bbox
     - floor_grating: plate-like AND low "solidity" vs solid plate of same thickness
     - hardware: small + light
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from step4_unknown_gallery import UnknownGalleryConfig, write_unknown_gallery


BUCKETS = [
    "plates",
    "sections",
    "hardware",
    "handrail",
    "stairtreads",
    "floor_grating",
    "unknown",
]

OUT_COLUMNS = [
    "def_id",
    "name",
    "qty_total",
    "shape_kind",
    "solid_count",
    "bbox_min",
    "bbox_max",
    "bbox_dx",
    "bbox_dy",
    "bbox_dz",
    # massprops + derived
    "area_mm2",
    "volume_mm3",
    "density_kg_m3",
    "mass_kg",
    "bbox_vol_mm3",
    "fill_ratio_bbox",
    "kg_per_m2_bbox",
    "solid_plate_kg_per_m2",
    "grating_solidity",
    # classification + manifest
    "bucket",
    "match_status",
    "stl_path",
    "part_id",
    "def_sig_used",
    "def_sig_source",
]

# Name-first keywords (conservative)
HW_KEYWORDS = [
    "bolt", "nut", "washer", "screw", "stud", "anchor", "rivet", "pin", "dowel",
    "clip", "clamp", "fastener", "fixing", "hinge", "latch", "shim", "spacer",
    "u-bolt", "ubolt", "thread", "rod", "nyloc", "spring washer",
]

HANDRAIL_KEYWORDS = [
    "handrail", "hand rail", "railing", "guardrail", "guard rail",
    "balustrade", "stanchion", "newel",
]

STAIRTREAD_KEYWORDS = [
    "stairtread", "stair tread", "tread", "stair step", "step tread",
]

FLOOR_GRATING_KEYWORDS = [
    "grating", "floor grating", "walkway grating", "mesh grating",
    "grate", "grille", "bar grating", "serrated grating",
]

PLATE_KEYWORDS = [
    "plate", "baseplate", "base plate", "gusset", "stiffener", "end plate",
    "splice plate", "flitch", "cleat", "bracket",
]

SECTION_KEYWORDS = [
    "beam", "column", "joist", "channel", "pfc", "ub", "uc", "ipe", "ipn", "hea", "heb",
    "upn", "unp", "upe", "rhs", "shs", "chs", "angle", "ea", "ua", "tee", "t-section",
    "flat bar", "round bar", "square bar",
]

RE_SECTION_DESIG = re.compile(
    r"\b(UB|UC|RHS|SHS|CHS|PFC|IPE|IPN|HEA|HEB|UNP|UPN|UPE|EA|UA)\b",
    flags=re.IGNORECASE,
)

RE_HARDWARE_DESIG = re.compile(
    r"\b(M\d{1,3}\b|M\d{1,3}x\d{1,4}\b|DIN\b|ISO\b|BS\b)\b",
    flags=re.IGNORECASE,
)

# Words that indicate fabricated steel parts (NOT bought-in hardware),
# even if the name also contains something like "M12" etc.
HARDWARE_EXCLUDE_KEYWORDS = [
    "plate", "baseplate", "base plate", "gusset", "stiffener", "end plate",
    "splice", "cleat", "bracket", "lug", "tab", "clip plate",
    "frame", "beam", "column", "channel", "angle", "rhs", "shs", "chs",
    "handrail", "railing", "tread", "grating",
]

def is_hardware_by_name(name: str) -> bool:
    """
    Hardware is only name/pattern based.
    Requires: (hardware keyword OR DIN/ISO/BS/Mxx pattern)
    AND must NOT look like fabricated steel (plates/brackets/cleats/etc).
    """
    n = _norm_name(name)

    looks_hardware = _has_any_kw(n, HW_KEYWORDS) or (RE_HARDWARE_DESIG.search(name or "") is not None)
    if not looks_hardware:
        return False

    looks_fabricated = _has_any_kw(n, HARDWARE_EXCLUDE_KEYWORDS) or (RE_SECTION_DESIG.search(name or "") is not None)
    if looks_fabricated:
        return False

    return True

@dataclass(frozen=True)
class PlateHeuristic:
    thick_max_mm: float
    thin_ratio: float
    min_span_mm: float


@dataclass(frozen=True)
class OtherHeuristics:
    section_len_ratio: float
    hardware_max_dim_mm: float
    hardware_max_mass_kg: float
    grating_solidity_max: float  # <= => grating-like


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def _norm_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _has_any_kw(s: str, kws: List[str]) -> bool:
    return any(kw in s for kw in kws)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _fmt_xyz(vals: Any) -> str:
    if not isinstance(vals, (list, tuple)) or len(vals) != 3:
        return ""
    try:
        return f"{float(vals[0]):.3f},{float(vals[1]):.3f},{float(vals[2]):.3f}"
    except Exception:
        return ""


def _bbox_from_def(defn: Dict[str, Any]) -> Tuple[str, str, float, float, float]:
    bbox = defn.get("bbox") or {}
    bmin = bbox.get("min")
    bmax = bbox.get("max")
    bmin_s = _fmt_xyz(bmin)
    bmax_s = _fmt_xyz(bmax)

    dx = dy = dz = 0.0
    if isinstance(bmin, (list, tuple)) and isinstance(bmax, (list, tuple)) and len(bmin) == 3 and len(bmax) == 3:
        x0, y0, z0 = (_safe_float(bmin[0]), _safe_float(bmin[1]), _safe_float(bmin[2]))
        x1, y1, z1 = (_safe_float(bmax[0]), _safe_float(bmax[1]), _safe_float(bmax[2]))
        if None not in (x0, y0, z0, x1, y1, z1):
            dx = max(0.0, x1 - x0)
            dy = max(0.0, y1 - y0)
            dz = max(0.0, z1 - z0)

    return bmin_s, bmax_s, float(dx), float(dy), float(dz)


def _massprops(defn: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    mp = defn.get("massprops") or {}
    area = _safe_float(mp.get("area"))           # mm^2
    vol = _safe_float(mp.get("volume"))          # mm^3
    dens = _safe_float(mp.get("density_kg_m3"))  # kg/m^3
    mass = _safe_float(mp.get("mass_kg"))        # kg
    return area, vol, dens, mass


def _bbox_volume_mm3(dx: float, dy: float, dz: float) -> float:
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return 0.0
    return float(dx) * float(dy) * float(dz)


def _fill_ratio_bbox(volume_mm3: Optional[float], bbox_vol_mm3: float) -> Optional[float]:
    if volume_mm3 is None or bbox_vol_mm3 <= 0.0:
        return None
    r = float(volume_mm3) / float(bbox_vol_mm3)
    if r < 0.0:
        r = 0.0
    if r > 1.5:
        r = 1.5
    return r


def _dims_sorted(dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
    dims = sorted([float(dx), float(dy), float(dz)])
    return dims[0], dims[1], dims[2]  # t, mid, L


def _is_plate_like(dx: float, dy: float, dz: float, h: PlateHeuristic) -> bool:
    t, mid, L = _dims_sorted(dx, dy, dz)
    if t <= 0.0 or mid <= 0.0 or L <= 0.0:
        return False
    if t > h.thick_max_mm:
        return False
    if mid < h.min_span_mm or L < h.min_span_mm:
        return False
    if t > (h.thin_ratio * mid):
        return False
    if t > (h.thin_ratio * L):
        return False
    return True


def _kg_per_m2_from_bbox_plan(mass_kg: Optional[float], mid_mm: float, L_mm: float) -> Optional[float]:
    if mass_kg is None:
        return None
    plan_area_mm2 = float(mid_mm) * float(L_mm)
    if plan_area_mm2 <= 0.0:
        return None
    return float(mass_kg) / (plan_area_mm2 / 1e6)


def _solid_plate_kg_per_m2(density_kg_m3: Optional[float], thickness_mm: float) -> Optional[float]:
    if density_kg_m3 is None or thickness_mm <= 0.0:
        return None
    return float(density_kg_m3) * (float(thickness_mm) / 1000.0)


def classify_name_first(name: str) -> str:
    n = _norm_name(name)

    if _has_any_kw(n, HANDRAIL_KEYWORDS):
        return "handrail"
    if _has_any_kw(n, STAIRTREAD_KEYWORDS):
        return "stairtreads"
    if _has_any_kw(n, FLOOR_GRATING_KEYWORDS):
        return "floor_grating"

    # Hardware is name/pattern-only, with a fabricated-parts exclusion guard
    if is_hardware_by_name(name):
        return "hardware"

    if _has_any_kw(n, PLATE_KEYWORDS):
        return "plates"
    if _has_any_kw(n, SECTION_KEYWORDS) or RE_SECTION_DESIG.search(name or ""):
        return "sections"

    return "unknown"

def classify_geometry_fallback(
    dx: float, dy: float, dz: float,
    plate_h: PlateHeuristic,
    other_h: OtherHeuristics,
    density_kg_m3: Optional[float],
    mass_kg: Optional[float],
) -> str:
    t, mid, L = _dims_sorted(dx, dy, dz)

    # sections: long slender (avoid plates)
    if mid > 0.0 and (L / mid) >= other_h.section_len_ratio:
        if not _is_plate_like(dx, dy, dz, plate_h):
            return "sections"

    # plates / grating: plate-like
    if _is_plate_like(dx, dy, dz, plate_h):
        kg_m2 = _kg_per_m2_from_bbox_plan(mass_kg, mid, L)
        solid_kg_m2 = _solid_plate_kg_per_m2(density_kg_m3, t)
        if kg_m2 is not None and solid_kg_m2 is not None and solid_kg_m2 > 0.0:
            solidity = kg_m2 / solid_kg_m2
            if solidity <= other_h.grating_solidity_max:
                return "floor_grating"
        return "plates"

    return "unknown"


def _pick_best_manifest_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    def score(it: Dict[str, Any]) -> Tuple[int, int]:
        ms = _as_str(it.get("match_status")).lower()
        stl = _as_str(it.get("stl_path")).strip()
        return (1 if ms == "matched" else 0, 1 if stl else 0)
    return sorted(items, key=score, reverse=True)[0]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {c: "" for c in OUT_COLUMNS}
            for c in OUT_COLUMNS:
                v = r.get(c, "")
                out[c] = "" if v is None else str(v)
            w.writerow(out)


def build_rows(xcaf: Dict[str, Any], stl_manifest: Dict[str, Any], plate_h: PlateHeuristic, other_h: OtherHeuristics) -> List[Dict[str, Any]]:
    defs = xcaf.get("definitions") or {}
    items = stl_manifest.get("items") or []

    by_def: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        ref_def = _as_str(it.get("ref_def")).strip()
        if ref_def:
            by_def.setdefault(ref_def, []).append(it)

    rows: List[Dict[str, Any]] = []

    for def_id, d in defs.items():
        if not isinstance(d, dict):
            continue
        if not bool(d.get("has_shape", False)):
            continue

        name = _as_str(d.get("name")).strip()
        shape_kind = _as_str(d.get("shape_kind")).strip()
        solid_count = d.get("solid_count", "")
        qty_total = d.get("qty_total", "")

        bmin_s, bmax_s, dx, dy, dz = _bbox_from_def(d)
        area, vol, dens, mass = _massprops(d)

        bbox_vol = _bbox_volume_mm3(dx, dy, dz)
        fill = _fill_ratio_bbox(vol, bbox_vol)

        t, mid, L = _dims_sorted(dx, dy, dz)
        kg_m2 = _kg_per_m2_from_bbox_plan(mass, mid, L)
        solid_kg_m2 = _solid_plate_kg_per_m2(dens, t)
        solidity = None
        if kg_m2 is not None and solid_kg_m2 is not None and solid_kg_m2 > 0.0:
            solidity = kg_m2 / solid_kg_m2

        bucket = classify_name_first(name)
        if bucket == "unknown":
            bucket = classify_geometry_fallback(dx, dy, dz, plate_h, other_h, dens, mass)

        best = _pick_best_manifest_item(by_def.get(str(def_id), []))
        match_status = _as_str(best.get("match_status")).strip() or "missing_manifest"

        rows.append({
            "def_id": str(def_id),
            "name": name,
            "qty_total": qty_total,
            "shape_kind": shape_kind,
            "solid_count": solid_count,
            "bbox_min": bmin_s,
            "bbox_max": bmax_s,
            "bbox_dx": f"{dx:.3f}",
            "bbox_dy": f"{dy:.3f}",
            "bbox_dz": f"{dz:.3f}",
            "area_mm2": f"{area:.3f}" if area is not None else "",
            "volume_mm3": f"{vol:.3f}" if vol is not None else "",
            "density_kg_m3": f"{dens:.3f}" if dens is not None else "",
            "mass_kg": f"{mass:.6f}" if mass is not None else "",
            "bbox_vol_mm3": f"{bbox_vol:.3f}" if bbox_vol > 0.0 else "",
            "fill_ratio_bbox": f"{fill:.6f}" if fill is not None else "",
            "kg_per_m2_bbox": f"{kg_m2:.6f}" if kg_m2 is not None else "",
            "solid_plate_kg_per_m2": f"{solid_kg_m2:.6f}" if solid_kg_m2 is not None else "",
            "grating_solidity": f"{solidity:.6f}" if solidity is not None else "",
            "bucket": bucket,
            "match_status": match_status,
            "stl_path": _as_str(best.get("stl_path")).strip(),
            "part_id": _as_str(best.get("part_id")).strip(),
            "def_sig_used": _as_str(best.get("def_sig_used")).strip(),
            "def_sig_source": _as_str(best.get("def_sig_source")).strip(),
        })

    rows.sort(key=lambda r: (r.get("bucket", ""), _norm_name(r.get("name", "")), r.get("def_id", "")))
    return rows

def write_outputs(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(out_dir / "step4_parts.csv", rows)
    for b in BUCKETS:
        sub = [r for r in rows if r.get("bucket") == b]
        _write_csv(out_dir / f"step4_proc_{b}.csv", sub)

    write_unknown_diagnostics(out_dir, rows, top_n=150)

def write_unknown_diagnostics(out_dir: Path, rows: List[Dict[str, Any]], top_n: int = 100) -> None:
    unk = [r for r in rows if r.get("bucket") == "unknown"]

    def fnum(s: str) -> float:
        try:
            return float(s)
        except Exception:
            return 0.0

    by_qty = sorted(unk, key=lambda r: fnum(_as_str(r.get("qty_total"))), reverse=True)[:top_n]
    by_mass = sorted(unk, key=lambda r: fnum(_as_str(r.get("mass_kg"))), reverse=True)[:top_n]

    _write_csv(out_dir / "step4_unknown_top_by_qty.csv", by_qty)
    _write_csv(out_dir / "step4_unknown_top_by_mass.csv", by_mass)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xcaf", default="/out/xcaf_instances.json")
    p.add_argument("--manifest", default="/out/stl_manifest.json")
    p.add_argument("--out-dir", default="/out")

    # plate
    p.add_argument("--plate-thick-max-mm", type=float, default=None)
    p.add_argument("--plate-thin-ratio", type=float, default=None)
    p.add_argument("--plate-min-span-mm", type=float, default=None)

    # geometry fallback
    p.add_argument("--section-len-ratio", type=float, default=None)
    p.add_argument("--hardware-max-dim-mm", type=float, default=None)
    p.add_argument("--hardware-max-mass-kg", type=float, default=None)
    p.add_argument("--grating-solidity-max", type=float, default=None)

    p.add_argument("--make-unknown-gallery", action="store_true",
               help="Write /out/step4_unknown_gallery (HTML + copied STLs) for unknown items.")
    p.add_argument("--unknown-gallery-max", type=int, default=None,
               help="Max number of unknowns to include in gallery (default env STEP4_UNKNOWN_GALLERY_MAX or 800).")
    p.add_argument("--unknown-gallery-only-with-stl", action="store_true",
               help="Only include unknowns that have an STL.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    xcaf_path = Path(args.xcaf)
    man_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if not xcaf_path.is_file():
        raise SystemExit(f"[step4] Missing xcaf file: {xcaf_path}")
    if not man_path.is_file():
        raise SystemExit(f"[step4] Missing manifest file: {man_path}")

    # env defaults
    thick_max = _env_float("STEP4_PLATE_THICK_MAX_MM", 30.0)
    thin_ratio = _env_float("STEP4_PLATE_THIN_RATIO", 0.10)
    min_span = _env_float("STEP4_PLATE_MIN_SPAN_MM", 50.0)

    section_len_ratio = _env_float("STEP4_SECTION_LEN_RATIO", 4.0)
    hardware_max_dim_mm = _env_float("STEP4_HARDWARE_MAX_DIM_MM", 200.0)
    hardware_max_mass_kg = _env_float("STEP4_HARDWARE_MAX_MASS_KG", 5.0)
    grating_solidity_max = _env_float("STEP4_GRATING_SOLIDITY_MAX", 0.55)

    # CLI overrides
    if args.plate_thick_max_mm is not None:
        thick_max = float(args.plate_thick_max_mm)
    if args.plate_thin_ratio is not None:
        thin_ratio = float(args.plate_thin_ratio)
    if args.plate_min_span_mm is not None:
        min_span = float(args.plate_min_span_mm)

    if args.section_len_ratio is not None:
        section_len_ratio = float(args.section_len_ratio)
    if args.hardware_max_dim_mm is not None:
        hardware_max_dim_mm = float(args.hardware_max_dim_mm)
    if args.hardware_max_mass_kg is not None:
        hardware_max_mass_kg = float(args.hardware_max_mass_kg)
    if args.grating_solidity_max is not None:
        grating_solidity_max = float(args.grating_solidity_max)

    plate_h = PlateHeuristic(thick_max_mm=thick_max, thin_ratio=thin_ratio, min_span_mm=min_span)
    other_h = OtherHeuristics(
        section_len_ratio=section_len_ratio,
        hardware_max_dim_mm=hardware_max_dim_mm,
        hardware_max_mass_kg=hardware_max_mass_kg,
        grating_solidity_max=grating_solidity_max,
    )

    xcaf = _load_json(xcaf_path)
    man = _load_json(man_path)

    rows = build_rows(xcaf, man, plate_h, other_h)
    write_outputs(out_dir, rows)
    # Unknown gallery (optional)
    make_gallery_env = os.getenv("STEP4_UNKNOWN_GALLERY", "").strip() in ("1", "true", "yes", "on")
    if args.make_unknown_gallery or make_gallery_env:
        max_env = int(os.getenv("STEP4_UNKNOWN_GALLERY_MAX", "800").strip() or "800")
        max_items = args.unknown_gallery_max if args.unknown_gallery_max is not None else max_env
        cfg = UnknownGalleryConfig(
            max_items=int(max_items),
            only_with_stl=bool(args.unknown_gallery_only_with_stl),
            prefer_with_stl=True,
        )
        gdir = write_unknown_gallery(out_dir, rows, cfg=cfg)
        print(f"[step4] Unknown gallery: {gdir}")


    counts: Dict[str, int] = {b: 0 for b in BUCKETS}
    for r in rows:
        counts[r.get("bucket", "unknown")] = counts.get(r.get("bucket", "unknown"), 0) + 1

    print("[step4] Wrote:", out_dir / "step4_parts.csv")
    for b in BUCKETS:
        print(f"[step4] {b:13s}: {counts.get(b, 0)}  -> {out_dir / f'step4_proc_{b}.csv'}")

    print(f"[step4] Plate: THICK_MAX={plate_h.thick_max_mm}  THIN_RATIO={plate_h.thin_ratio}  MIN_SPAN={plate_h.min_span_mm}")
    print(f"[step4] Sections: LEN_RATIO={other_h.section_len_ratio}")
    print(f"[step4] Hardware: MAX_DIM={other_h.hardware_max_dim_mm}  MAX_MASS={other_h.hardware_max_mass_kg}")
    print(f"[step4] Grating: SOLIDITY_MAX={other_h.grating_solidity_max} (actual kg/m2 / solid plate kg/m2)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Step 4 - Multi-body Review rules (Stage 1–3 only)

Pure logic + constants:
- conservative auto_keep thresholds
- likely_explode heuristics (still only suggests)
- short keyword list
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ----------------------------
# Conservative thresholds
# ----------------------------

AUTO_KEEP_SOLIDCOUNT_MIN: int = 20
AUTO_KEEP_QTY_MIN: int = 50
AUTO_KEEP_SOLIDCOUNT_FOR_HIGH_QTY_MIN: int = 6

LIKELY_EXPLODE_SOLIDCOUNT_MAX: int = 4
LIKELY_EXPLODE_QTY_MAX: int = 10


# ----------------------------
# Short keyword list (case-insensitive substring match)
# Keep it deliberately small at first.
# ----------------------------

AUTO_KEEP_KEYWORDS = [
    "grating",
    "handrail",
    "ladder",
    "stair",
    "platform",
    "cage",
    "guard",
    "toe board",
    "kick plate",
    "vendor",
    "assembly",
    "ass'y",
    "sub-assy",
    "fixing",
    "fastener",
    "bolt",
    "nut",
    "washer",
    "anchor",
    "hilti",
    "unistrut",
    "pipe clamp",
]


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    reason: str


def _name_has_autokeep_keyword(name: str) -> Tuple[bool, str]:
    n = (name or "").strip().lower()
    for kw in AUTO_KEEP_KEYWORDS:
        if kw in n:
            return True, kw
    return False, ""


def bucket_candidate(name: str, qty_total: int, solid_count: int) -> BucketResult:
    """
    Returns (bucket, reason) where bucket in:
      - auto_keep
      - review
      - likely_explode

    This does NOT explode anything; it's just bucketing for review output.
    """
    # Guardrails
    if solid_count < 2:
        return BucketResult(bucket="review", reason="not multi-body (solid_count<2)")

    has_kw, kw = _name_has_autokeep_keyword(name)

    if solid_count >= AUTO_KEEP_SOLIDCOUNT_MIN:
        return BucketResult(
            bucket="auto_keep",
            reason=f"solid_count>={AUTO_KEEP_SOLIDCOUNT_MIN} (likely vendor assembly/grating)",
        )

    if qty_total >= AUTO_KEEP_QTY_MIN and solid_count >= AUTO_KEEP_SOLIDCOUNT_FOR_HIGH_QTY_MIN:
        return BucketResult(
            bucket="auto_keep",
            reason=f"qty_total>={AUTO_KEEP_QTY_MIN} + solid_count>={AUTO_KEEP_SOLIDCOUNT_FOR_HIGH_QTY_MIN} (likely standard assembly)",
        )

    if has_kw:
        return BucketResult(bucket="auto_keep", reason=f"name keyword '{kw}'")

    # Suggest likely explode only for small packs and low qty
    if solid_count <= LIKELY_EXPLODE_SOLIDCOUNT_MAX and qty_total <= LIKELY_EXPLODE_QTY_MAX:
        return BucketResult(
            bucket="likely_explode",
            reason=f"small solid_count (<= {LIKELY_EXPLODE_SOLIDCOUNT_MAX}) + low qty (<= {LIKELY_EXPLODE_QTY_MAX})",
        )

    return BucketResult(bucket="review", reason="default review")

# brep_signature.py
# Deterministic, quantized B-Rep feature signatures (OCP).
#
# v5 goals:
#   - Reduce def_sig collisions for parts that differ only by hole patterns/locations.
#   - Add location-sensitive cylindrical feature descriptors in a deterministic local frame.
#   - Keep compute_def_sig_free mirror-invariant by design.
#   - Power-of-10 style: bounded loops, guards, deterministic ordering.
#
# IMPORTANT:
#   - All signature logic must live here. Step 1/2 import only:
#       from brep_signature import DEF_SIG_ALGO, compute_def_sig, compute_def_sig_free
#
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as _np

from OCP.GProp import GProp_GProps
from OCP.TopAbs import (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_WIRE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPSOLID,
    TopAbs_COMPOUND,
)
from OCP.TopExp import TopExp_Explorer
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Tool
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve,
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
)

import OCP.BRepGProp as BRepGProp
import OCP.BRepBndLib as BRepBndLib


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Bumped because payload changed (location-sensitive features added).
DEF_SIG_ALGO = "brep_features_v5_ocp_sha256"


def call_maybe_s(obj, method: str, *args):
    """
    Call obj.method(*args) or obj.method_s(*args) for OCP binding variants.
    """
    fn = getattr(obj, method, None)
    if callable(fn):
        try:
            return fn(*args)
        except TypeError:
            pass
    fn_s = getattr(obj, method + "_s", None)
    if callable(fn_s):
        return fn_s(*args)
    raise AttributeError(f"{type(obj).__name__} has no usable {method} / {method}_s")


def compute_def_sig(shape) -> str:
    """
    Mirror-sensitive where possible:
      - signed normal bins
      - signed local-frame position bins
      - signed local-frame axis direction bins (canonicalized sign)
    """
    payload = _compute_feature_payload(shape, mirror_invariant=False)
    return _hash_payload(payload)


def compute_def_sig_free(shape) -> str:
    """
    Mirror-invariant by design:
      - normals use abs components
      - local-frame positions use abs coordinates
      - axis direction bins use abs components
    """
    payload = _compute_feature_payload(shape, mirror_invariant=True)
    return _hash_payload(payload)


# ---------------------------------------------------------------------------
# Quantization + guards (mm units)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Q:
    # Scalars
    VOL_STEP: float = 0.1          # mm^3
    AREA_STEP: float = 0.1         # mm^2
    MOM_STEP: float = 0.01         # inertia moments (quantized)
    LEN_STEP: float = 0.1          # mm (edge length, cylinder length)
    RAD_STEP: float = 0.1          # mm (radii)
    ANG_STEP_DEG: float = 0.25     # degrees
    FRAC_STEP: float = 1e-4        # [0..1] bins
    NORM_STEP: float = 0.1         # unit vector component step
    POS_STEP: float = 0.25         # mm (projected cylinder axis position)
    # Positional histogram grid
    GRID_N: int = 8                # <= 8, bounded
    # Debug
    DEBUG_TOP_N: int = 12
    # Hard caps
    MAX_FACES: int = 200000
    MAX_EDGES: int = 400000
    MAX_VERTS: int = 800000
    MAX_SIGN_VERTS: int = 5000     # for frame sign determination
    MAX_CYL_FACES: int = 200000    # still bounded by MAX_FACES


_Q = Q()


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------

def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_debug() -> bool:
    return str(os.environ.get("SIG_DEBUG", "")).strip() == "1"


def _topn(counter: Counter, n: int) -> List[Tuple[Any, int]]:
    return counter.most_common(int(n))


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _q_int(v: float, step: float) -> int:
    if step <= 0:
        return int(round(float(v)))
    return int(round(float(v) / float(step)))


def _q_frac(v: float) -> int:
    if not math.isfinite(v) or v < 0:
        return 0
    if v > 1:
        v = 1.0
    return _q_int(v, _Q.FRAC_STEP)


def _q_deg(v_deg: float) -> int:
    if not math.isfinite(v_deg):
        return 0
    return _q_int(v_deg, _Q.ANG_STEP_DEG)


def _q_norm3(nx: float, ny: float, nz: float, *, mirror_invariant: bool) -> Tuple[int, int, int]:
    if mirror_invariant:
        nx, ny, nz = abs(nx), abs(ny), abs(nz)
    return (_q_int(nx, _Q.NORM_STEP), _q_int(ny, _Q.NORM_STEP), _q_int(nz, _Q.NORM_STEP))


def _q_pos3(x: float, y: float, z: float, *, mirror_invariant: bool) -> Tuple[int, int, int]:
    if mirror_invariant:
        x, y, z = abs(x), abs(y), abs(z)
    return (_q_int(x, _Q.POS_STEP), _q_int(y, _Q.POS_STEP), _q_int(z, _Q.POS_STEP))


def _encode_tuple_key(k: Any) -> str:
    if isinstance(k, tuple):
        return ",".join(str(int(x)) for x in k)
    return str(k)


def _counter_to_sorted_dict(counter: Counter, *, key_encoder=None) -> Dict[str, int]:
    if not counter:
        return {}
    enc = key_encoder if key_encoder is not None else (lambda x: str(x))
    items = [(enc(k), int(v)) for k, v in counter.items()]
    items.sort(key=lambda kv: kv[0])
    return {k: v for (k, v) in items}


# ---------------------------------------------------------------------------
# Basic shape properties / topology
# ---------------------------------------------------------------------------

def _count_topology(shape) -> Dict[str, int]:
    def count_of(kind) -> int:
        it = TopExp_Explorer(shape, kind)
        c = 0
        while it.More():
            c += 1
            it.Next()
        return c

    return {
        "n_vertex": count_of(TopAbs_VERTEX),
        "n_edge": count_of(TopAbs_EDGE),
        "n_wire": count_of(TopAbs_WIRE),
        "n_face": count_of(TopAbs_FACE),
        "n_shell": count_of(TopAbs_SHELL),
        "n_solid": count_of(TopAbs_SOLID),
        "n_compsolid": count_of(TopAbs_COMPSOLID),
        "n_compound": count_of(TopAbs_COMPOUND),
    }


def _volume_area_centroid_inertia(shape) -> Tuple[float, float, Tuple[float, float, float], _np.ndarray]:
    """
    Returns:
      volume (float), area (float),
      centroid (x,y,z),
      inertia 3x3 matrix about origin (from props_v.MatrixOfInertia when available)
    """
    props_v = GProp_GProps()
    call_maybe_s(BRepGProp, "VolumeProperties", shape, props_v)
    vol = float(props_v.Mass())

    props_a = GProp_GProps()
    call_maybe_s(BRepGProp, "SurfaceProperties", shape, props_a)
    area = float(props_a.Mass())

    try:
        c = props_v.CentreOfMass()
        cx, cy, cz = float(c.X()), float(c.Y()), float(c.Z())
    except Exception:
        cx, cy, cz = 0.0, 0.0, 0.0

    # Inertia matrix (symmetric). OCP binding names may vary.
    I = _np.zeros((3, 3), dtype=float)
    try:
        m = props_v.MatrixOfInertia()
        # gp_Mat supports Value(row,col) 1-based
        I[0, 0] = float(m.Value(1, 1))
        I[0, 1] = float(m.Value(1, 2))
        I[0, 2] = float(m.Value(1, 3))
        I[1, 0] = float(m.Value(2, 1))
        I[1, 1] = float(m.Value(2, 2))
        I[1, 2] = float(m.Value(2, 3))
        I[2, 0] = float(m.Value(3, 1))
        I[2, 1] = float(m.Value(3, 2))
        I[2, 2] = float(m.Value(3, 3))
    except Exception:
        # Fallback: try PrincipalProperties moments only -> diagonal-ish (weak)
        try:
            pp = props_v.PrincipalProperties()
            I1, I2, I3 = pp.Moments()
            I[0, 0], I[1, 1], I[2, 2] = float(I1), float(I2), float(I3)
        except Exception:
            pass

    return vol, area, (cx, cy, cz), I


def _volume_area_moments_quant(shape) -> Tuple[int, int, Tuple[int, int, int], Tuple[int, int]]:
    props_v = GProp_GProps()
    call_maybe_s(BRepGProp, "VolumeProperties", shape, props_v)
    vol = float(props_v.Mass())

    props_a = GProp_GProps()
    call_maybe_s(BRepGProp, "SurfaceProperties", shape, props_a)
    area = float(props_a.Mass())

    moms = [0.0, 0.0, 0.0]
    try:
        I1, I2, I3 = props_v.PrincipalProperties().Moments()
        moms = [float(I1), float(I2), float(I3)]
    except Exception:
        pass
    moms.sort()

    r12 = (moms[1] / moms[0]) if moms[0] > 0 else 0.0
    r23 = (moms[2] / moms[1]) if moms[1] > 0 else 0.0

    vol_q = _q_int(vol, _Q.VOL_STEP)
    area_q = _q_int(area, _Q.AREA_STEP)
    moms_q = (_q_int(moms[0], _Q.MOM_STEP), _q_int(moms[1], _Q.MOM_STEP), _q_int(moms[2], _Q.MOM_STEP))
    ratios_q = (_q_int(r12, 1e-4), _q_int(r23, 1e-4))
    return vol_q, area_q, moms_q, ratios_q


def _aabb_extents_sorted(shape) -> Tuple[int, int, int]:
    box = Bnd_Box()
    # OCP variant: Add vs Add_s
    call_maybe_s(BRepBndLib, "Add", shape, box)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except TypeError:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get_s()

    ex = max(0.0, float(xmax - xmin))
    ey = max(0.0, float(ymax - ymin))
    ez = max(0.0, float(zmax - zmin))

    exi, eyi, ezi = _q_int(ex, _Q.LEN_STEP), _q_int(ey, _Q.LEN_STEP), _q_int(ez, _Q.LEN_STEP)
    vals = sorted([exi, eyi, ezi], reverse=True)
    return (vals[0], vals[1], vals[2])


def _face_area(face) -> float:
    props = GProp_GProps()
    call_maybe_s(BRepGProp, "SurfaceProperties", face, props)
    return float(props.Mass())


def _edge_length(edge) -> float:
    props = GProp_GProps()
    call_maybe_s(BRepGProp, "LinearProperties", edge, props)
    return float(props.Mass())


# ---------------------------------------------------------------------------
# Deterministic local frame (principal-ish) + sign canonicalization
# ---------------------------------------------------------------------------

def _iter_vertex_points(shape, max_count: int) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    it = TopExp_Explorer(shape, TopAbs_VERTEX)
    while it.More() and len(pts) < max_count:
        v = it.Current()
        try:
            p = BRep_Tool.Pnt(v)
            pts.append((float(p.X()), float(p.Y()), float(p.Z())))
        except Exception:
            pass
        it.Next()
    return pts


def _normalize(v: _np.ndarray) -> _np.ndarray:
    n = float(_np.linalg.norm(v))
    if n <= 1e-30:
        return v * 0.0
    return v / n


def _canonicalize_axis_sign(axis: _np.ndarray) -> _np.ndarray:
    """
    Deterministic sign for a single axis vector:
      - make its largest-magnitude component positive.
    """
    a = axis.copy()
    idx = int(_np.argmax(_np.abs(a)))
    if a[idx] < 0:
        a = -a
    return a


def _make_right_handed(u: _np.ndarray, v: _np.ndarray, w_hint: _np.ndarray) -> Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Ensure (u,v,w) is right-handed, with w aligned (as much as possible) to w_hint.
    """
    u = _normalize(u)
    v = _normalize(v - u * float(_np.dot(u, v)))
    w = _normalize(_np.cross(u, v))
    if float(_np.dot(w, w_hint)) < 0:
        v = -v
        w = -w
    return u, v, w


def _frame_from_inertia(shape) -> Tuple[Tuple[float, float, float], _np.ndarray, _np.ndarray, _np.ndarray, Dict[str, Any]]:
    """
    Build a deterministic local frame from inertia eigenvectors, with stable ordering + signs.

    Returns:
      origin (centroid),
      u,v,w unit vectors (np arrays),
      diag info dict for debugging.
    """
    vol, area, centroid, I = _volume_area_centroid_inertia(shape)
    cx, cy, cz = centroid

    # Eigenvectors of symmetric matrix
    try:
        evals, evecs = _np.linalg.eigh(I)  # sorted ascending by eval
    except Exception:
        evals = _np.array([0.0, 0.0, 0.0], dtype=float)
        evecs = _np.eye(3, dtype=float)

    # Candidate axes are columns of evecs
    axes = [evecs[:, 0].copy(), evecs[:, 1].copy(), evecs[:, 2].copy()]

    # If matrix was junk, fallback to world axes
    for i in range(3):
        if float(_np.linalg.norm(axes[i])) <= 1e-30:
            axes[i] = _np.array([1.0, 0.0, 0.0], dtype=float) if i == 0 else (
                _np.array([0.0, 1.0, 0.0], dtype=float) if i == 1 else _np.array([0.0, 0.0, 1.0], dtype=float)
            )

    # Compute extents along each axis using a bounded set of vertices
    pts = _iter_vertex_points(shape, _Q.MAX_SIGN_VERTS)
    if not pts:
        pts = [(cx, cy, cz)]

    P = _np.array(pts, dtype=float)
    C = _np.array([cx, cy, cz], dtype=float)
    V = P - C

    extents = []
    for a in axes:
        a = _normalize(a)
        proj = V @ a
        pmin = float(_np.min(proj))
        pmax = float(_np.max(proj))
        ext = pmax - pmin
        extents.append((ext, pmin, pmax))

    # Order axes by decreasing extent (more stable than raw eigenvalue order).
    order = list(range(3))
    order.sort(key=lambda i: (-extents[i][0], -abs(extents[i][2]), -abs(extents[i][1])))

    u0 = _normalize(axes[order[0]])
    v0 = _normalize(axes[order[1]])
    w0 = _normalize(axes[order[2]])

    # Canonical sign for each axis (largest component positive)
    u0 = _canonicalize_axis_sign(u0)
    v0 = _canonicalize_axis_sign(v0)
    w0 = _canonicalize_axis_sign(w0)

    # Make right-handed; align w with w0
    u, v, w = _make_right_handed(u0, v0, w0)

    diag = {
        "evals": [float(evals[0]), float(evals[1]), float(evals[2])],
        "extent_u": float(extents[order[0]][0]),
        "extent_v": float(extents[order[1]][0]),
        "extent_w": float(extents[order[2]][0]),
    }
    return (cx, cy, cz), u, v, w, diag


def _to_local(p: Tuple[float, float, float], origin: Tuple[float, float, float], u: _np.ndarray, v: _np.ndarray, w: _np.ndarray) -> Tuple[float, float, float]:
    px, py, pz = p
    ox, oy, oz = origin
    d = _np.array([px - ox, py - oy, pz - oz], dtype=float)
    return (float(_np.dot(d, u)), float(_np.dot(d, v)), float(_np.dot(d, w)))


def _dir_to_local(dxyz: Tuple[float, float, float], u: _np.ndarray, v: _np.ndarray, w: _np.ndarray) -> Tuple[float, float, float]:
    dx, dy, dz = dxyz
    d = _np.array([dx, dy, dz], dtype=float)
    dl = (_np.dot(d, u), _np.dot(d, v), _np.dot(d, w))
    # Normalize
    n = math.sqrt(float(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]))
    if n <= 1e-30:
        return (0.0, 0.0, 0.0)
    return (float(dl[0] / n), float(dl[1] / n), float(dl[2] / n))


def _canon_dir_local(dl: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Canonicalize direction sign: make the largest-magnitude component positive.
    """
    x, y, z = dl
    a = [x, y, z]
    idx = 0
    if abs(a[1]) > abs(a[idx]):
        idx = 1
    if abs(a[2]) > abs(a[idx]):
        idx = 2
    if a[idx] < 0:
        return (-x, -y, -z)
    return (x, y, z)


# ---------------------------------------------------------------------------
# Surface/curve type naming
# ---------------------------------------------------------------------------

def _surface_type_name(stype: int) -> str:
    if stype == GeomAbs_Plane:
        return "plane"
    if stype == GeomAbs_Cylinder:
        return "cylinder"
    if stype == GeomAbs_Cone:
        return "cone"
    if stype == GeomAbs_Sphere:
        return "sphere"
    if stype == GeomAbs_Torus:
        return "torus"
    if stype == GeomAbs_BSplineSurface:
        return "bspline"
    if stype == GeomAbs_BezierSurface:
        return "bezier"
    if stype == GeomAbs_SurfaceOfRevolution:
        return "revolution"
    if stype == GeomAbs_SurfaceOfExtrusion:
        return "extrusion"
    if stype == GeomAbs_OffsetSurface:
        return "offset"
    if stype == GeomAbs_OtherSurface:
        return "other"
    return "unknown"


def _curve_type_name(ctype: int) -> str:
    if ctype == GeomAbs_Line:
        return "line"
    if ctype == GeomAbs_Circle:
        return "circle"
    if ctype == GeomAbs_Ellipse:
        return "ellipse"
    if ctype == GeomAbs_Hyperbola:
        return "hyperbola"
    if ctype == GeomAbs_Parabola:
        return "parabola"
    if ctype == GeomAbs_BSplineCurve:
        return "bspline"
    if ctype == GeomAbs_BezierCurve:
        return "bezier"
    if ctype == GeomAbs_OffsetCurve:
        return "offset"
    if ctype == GeomAbs_OtherCurve:
        return "other"
    return "unknown"


# ---------------------------------------------------------------------------
# Face normal histogram (best-effort)
# ---------------------------------------------------------------------------

def _face_normal_quantized(face, *, mirror_invariant: bool) -> Optional[Tuple[int, int, int]]:
    try:
        s = BRepAdaptor_Surface(face)
        u0 = (s.FirstUParameter() + s.LastUParameter()) * 0.5
        v0 = (s.FirstVParameter() + s.LastVParameter()) * 0.5
        p, du, dv = s.D1(u0, v0)
        nx = du.Y() * dv.Z() - du.Z() * dv.Y()
        ny = du.Z() * dv.X() - du.X() * dv.Z()
        nz = du.X() * dv.Y() - du.Y() * dv.X()
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nlen <= 1e-30:
            return None
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
        return _q_norm3(nx, ny, nz, mirror_invariant=mirror_invariant)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# New: cylindrical features + planar inner-wire counts + positional bins
# ---------------------------------------------------------------------------

def _plane_inner_wire_count(face) -> int:
    """
    Count inner wires (holes) on a planar face.
    Deterministic and bounded.
    """
    try:
        outer = BRepTools.OuterWire(face)
    except Exception:
        outer = None

    cnt = 0
    itw = TopExp_Explorer(face, TopAbs_WIRE)
    while itw.More():
        w = itw.Current()
        try:
            is_outer = False
            if outer is not None:
                try:
                    is_outer = bool(w.IsSame(outer))
                except Exception:
                    is_outer = False
            if not is_outer:
                cnt += 1
        except Exception:
            pass
        itw.Next()

    # Guard: cap to a sane range (still deterministic)
    if cnt < 0:
        cnt = 0
    if cnt > 9999:
        cnt = 9999
    return cnt


def _cylinder_length_from_uv(face) -> float:
    """
    For GeomAbs_Cylinder, V parameter usually corresponds to axis direction (height).
    Use UV bounds if available.
    """
    try:
        umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    except Exception:
        try:
            umin, umax, vmin, vmax = BRepTools.UVBounds(face)
        except Exception:
            return 0.0
    return float(abs(vmax - vmin))


def _compute_cylinder_features(
    shape,
    origin: Tuple[float, float, float],
    u: _np.ndarray,
    v: _np.ndarray,
    w: _np.ndarray,
    *,
    mirror_invariant: bool,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """
    Returns:
      cyl_desc_hist: histogram of per-cylinder descriptors (string key -> count)
      cyl_r_hist: histogram of radius bins
      cyl_pos_bins: sparse 3D grid bins (i,j,k -> count)
      n_cyl_faces: number of cylindrical faces detected
    """
    cyl_desc_hist = Counter()
    cyl_r_hist = Counter()
    pos_bins = Counter()

    # For positional binning, we need extents in local frame.
    # Use a bounded set of vertices for extents (consistent with frame calc).
    pts = _iter_vertex_points(shape, _Q.MAX_SIGN_VERTS)
    if not pts:
        pts = [origin]

    # Compute local coords for extents
    locs = [_to_local(p, origin, u, v, w) for p in pts]

    if mirror_invariant:
        # abs extents [0..maxAbs]
        maxx = max(abs(x) for (x, _, _) in locs) if locs else 0.0
        maxy = max(abs(y) for (_, y, _) in locs) if locs else 0.0
        maxz = max(abs(z) for (_, _, z) in locs) if locs else 0.0
        minx, miny, minz = 0.0, 0.0, 0.0
        maxx, maxy, maxz = max(maxx, 1e-9), max(maxy, 1e-9), max(maxz, 1e-9)
    else:
        xs = [x for (x, _, _) in locs]
        ys = [y for (_, y, _) in locs]
        zs = [z for (_, _, z) in locs]
        minx, maxx = (min(xs), max(xs)) if xs else (0.0, 1e-9)
        miny, maxy = (min(ys), max(ys)) if ys else (0.0, 1e-9)
        minz, maxz = (min(zs), max(zs)) if zs else (0.0, 1e-9)
        if abs(maxx - minx) < 1e-9:
            maxx = minx + 1e-9
        if abs(maxy - miny) < 1e-9:
            maxy = miny + 1e-9
        if abs(maxz - minz) < 1e-9:
            maxz = minz + 1e-9

    n_cyl_faces = 0
    itf = TopExp_Explorer(shape, TopAbs_FACE)
    while itf.More():
        face = itf.Current()
        try:
            s = BRepAdaptor_Surface(face)
            st = s.GetType()
        except Exception:
            st = -1

        if st == GeomAbs_Cylinder:
            n_cyl_faces += 1

            try:
                cyl = s.Cylinder()
                r = float(cyl.Radius())
                # Axis location + direction in world
                ax1 = cyl.Axis()
                p0 = ax1.Location()
                d0 = ax1.Direction()
                p = (float(p0.X()), float(p0.Y()), float(p0.Z()))
                d = (float(d0.X()), float(d0.Y()), float(d0.Z()))
            except Exception:
                itf.Next()
                continue

            # Local frame coords
            pl = _to_local(p, origin, u, v, w)
            dl = _dir_to_local(d, u, v, w)
            dl = _canon_dir_local(dl)

            # Mirror invariance handled by quantizers
            rq = _q_int(r, _Q.RAD_STEP)
            pq = _q_pos3(pl[0], pl[1], pl[2], mirror_invariant=mirror_invariant)
            dq = _q_norm3(dl[0], dl[1], dl[2], mirror_invariant=mirror_invariant)

            # Length estimate
            L = _cylinder_length_from_uv(face)
            Lq = _q_int(L, _Q.LEN_STEP)

            # Descriptor key (deterministic string)
            # Includes radius + dir + pos + length
            key = f"r{rq}|d{dq[0]},{dq[1]},{dq[2]}|p{pq[0]},{pq[1]},{pq[2]}|L{Lq}"
            cyl_desc_hist[key] += 1
            cyl_r_hist[rq] += 1

            # Positional binning: map to GRID_N in each axis
            gx = (abs(pl[0]) if mirror_invariant else pl[0])
            gy = (abs(pl[1]) if mirror_invariant else pl[1])
            gz = (abs(pl[2]) if mirror_invariant else pl[2])

            nx = (gx - minx) / (maxx - minx)
            ny = (gy - miny) / (maxy - miny)
            nz = (gz - minz) / (maxz - minz)
            # clamp
            nx = 0.0 if nx < 0.0 else (1.0 if nx > 1.0 else nx)
            ny = 0.0 if ny < 0.0 else (1.0 if ny > 1.0 else ny)
            nz = 0.0 if nz < 0.0 else (1.0 if nz > 1.0 else nz)

            # bin index in [0..GRID_N-1]
            N = int(_Q.GRID_N)
            ix = int(min(N - 1, math.floor(nx * N)))
            iy = int(min(N - 1, math.floor(ny * N)))
            iz = int(min(N - 1, math.floor(nz * N)))
            pos_bins[(ix, iy, iz)] += 1

        itf.Next()

        # Guard: bounded traversal (already bounded by MAX_FACES, but keep explicit)
        if n_cyl_faces > _Q.MAX_CYL_FACES:
            break

    return (
        _counter_to_sorted_dict(cyl_desc_hist),
        _counter_to_sorted_dict(cyl_r_hist),
        _counter_to_sorted_dict(pos_bins, key_encoder=_encode_tuple_key),
        int(n_cyl_faces),
    )


# ---------------------------------------------------------------------------
# Main payload builder
# ---------------------------------------------------------------------------

def _compute_feature_payload(shape, *, mirror_invariant: bool) -> Dict[str, Any]:
    topo = _count_topology(shape)

    if topo["n_face"] > _Q.MAX_FACES or topo["n_edge"] > _Q.MAX_EDGES or topo["n_vertex"] > _Q.MAX_VERTS:
        return {
            "algo": DEF_SIG_ALGO,
            "abort": {
                "reason": "topology_limits",
                "n_face": topo["n_face"],
                "n_edge": topo["n_edge"],
                "n_vertex": topo["n_vertex"],
            },
            "topo": topo,
        }

    vol_q, area_q, moms_q, ratios_q = _volume_area_moments_quant(shape)
    ext_q = _aabb_extents_sorted(shape)
    chi = topo["n_vertex"] - topo["n_edge"] + topo["n_face"]

    # Stable local frame (principal-ish)
    origin, u, v, w, frame_diag = _frame_from_inertia(shape)

    # Face features (existing + planar inner wires)
    face_type_counts = Counter()
    face_area_frac_hist = Counter()
    face_norm_hist = Counter()
    face_area_hist = Counter()

    cyl_r_hist = Counter()
    cone_ang_hist = Counter()
    sphere_r_hist = Counter()
    torus_rr_hist = Counter()

    plane_inner_wire_hist = Counter()

    # Total area approx from quantized area (for stable fraction binning)
    total_area = max(1e-30, float(area_q) * _Q.AREA_STEP)

    itf = TopExp_Explorer(shape, TopAbs_FACE)
    while itf.More():
        face = itf.Current()
        try:
            s = BRepAdaptor_Surface(face)
            st = int(s.GetType())
        except Exception:
            st = -1

        st_name = _surface_type_name(st)
        face_type_counts[st_name] += 1

        # area bins
        try:
            a = _face_area(face)
        except Exception:
            a = 0.0
        face_area_hist[_q_int(a, _Q.AREA_STEP)] += 1

        frac = a / total_area if total_area > 0 else 0.0
        face_area_frac_hist[_q_frac(frac)] += 1

        nbin = _face_normal_quantized(face, mirror_invariant=mirror_invariant)
        if nbin is not None:
            face_norm_hist[nbin] += 1

        # planar inner wire counts
        if st == GeomAbs_Plane:
            plane_inner_wire_hist[_plane_inner_wire_count(face)] += 1

        # type-specific parameters
        try:
            if st == GeomAbs_Cylinder:
                r = float(s.Cylinder().Radius())
                cyl_r_hist[_q_int(r, _Q.RAD_STEP)] += 1
            elif st == GeomAbs_Cone:
                ang = float(s.Cone().SemiAngle())  # radians
                cone_ang_hist[_q_deg(math.degrees(ang))] += 1
            elif st == GeomAbs_Sphere:
                r = float(s.Sphere().Radius())
                sphere_r_hist[_q_int(r, _Q.RAD_STEP)] += 1
            elif st == GeomAbs_Torus:
                R = float(s.Torus().MajorRadius())
                r = float(s.Torus().MinorRadius())
                key = f"{_q_int(R, _Q.RAD_STEP)}:{_q_int(r, _Q.RAD_STEP)}"
                torus_rr_hist[key] += 1
        except Exception:
            pass

        itf.Next()

    # Edge features (existing)
    edge_type_counts = Counter()
    edge_len_hist = Counter()
    circle_r_hist = Counter()

    ite = TopExp_Explorer(shape, TopAbs_EDGE)
    while ite.More():
        edge = ite.Current()
        try:
            c = BRepAdaptor_Curve(edge)
            ct = int(c.GetType())
        except Exception:
            ct = -1

        edge_type_counts[_curve_type_name(ct)] += 1

        try:
            L = _edge_length(edge)
        except Exception:
            L = 0.0
        edge_len_hist[_q_int(L, _Q.LEN_STEP)] += 1

        try:
            if ct == GeomAbs_Circle:
                r = float(c.Circle().Radius())
                circle_r_hist[_q_int(r, _Q.RAD_STEP)] += 1
        except Exception:
            pass

        ite.Next()

    # New cylinder descriptors + positional bins in stable frame
    cyl_desc_hist, cyl_r_hist2, cyl_pos_bins, n_cyl_faces = _compute_cylinder_features(
        shape, origin, u, v, w, mirror_invariant=mirror_invariant
    )

    payload: Dict[str, Any] = {
        "algo": DEF_SIG_ALGO,
        "topo": topo,
        "props": {
            "vol_q": int(vol_q),
            "area_q": int(area_q),
            "moms_q": [int(moms_q[0]), int(moms_q[1]), int(moms_q[2])],
            "mom_ratio_q": [int(ratios_q[0]), int(ratios_q[1])],
            "aabb_ext_q_sorted": [int(ext_q[0]), int(ext_q[1]), int(ext_q[2])],
            "chi": int(chi),
            # Frame diagnostics included (quantized-ish) for stability/debug.
            # Not huge, deterministic.
            "frame_diag": frame_diag,
        },
        "faces": {
            "type_counts": _counter_to_sorted_dict(face_type_counts),
            "area_q_hist": _counter_to_sorted_dict(face_area_hist),
            "area_frac_hist": _counter_to_sorted_dict(face_area_frac_hist),
            "norm_hist": _counter_to_sorted_dict(face_norm_hist, key_encoder=_encode_tuple_key),
            "cyl_r_hist": _counter_to_sorted_dict(cyl_r_hist),
            "cone_ang_hist": _counter_to_sorted_dict(cone_ang_hist),
            "sphere_r_hist": _counter_to_sorted_dict(sphere_r_hist),
            "torus_rr_hist": _counter_to_sorted_dict(torus_rr_hist),
            "plane_inner_wires_hist": _counter_to_sorted_dict(plane_inner_wire_hist),
        },
        "edges": {
            "type_counts": _counter_to_sorted_dict(edge_type_counts),
            "len_q_hist": _counter_to_sorted_dict(edge_len_hist),
            "circle_r_hist": _counter_to_sorted_dict(circle_r_hist),
        },
        "cylinders": {
            "n_cyl_faces": int(n_cyl_faces),
            # Redundant but useful: radius bins from descriptor pass
            "r_hist": cyl_r_hist2,
            # Location-sensitive descriptors (most collision-breaking)
            "desc_hist": cyl_desc_hist,
            # Coarse position bins (<= 8^3 keys, sparse)
            "pos_bins": cyl_pos_bins,
        },
    }

    if _is_debug():
        _debug_print_summary(payload)

    return payload


def _debug_print_summary(payload: Dict[str, Any]) -> None:
    try:
        if "abort" in payload:
            print("[SIG_DEBUG] algo:", payload.get("algo"))
            print("[SIG_DEBUG] ABORT:", payload["abort"])
            return

        topo = payload.get("topo", {})
        props = payload.get("props", {})
        cyl = payload.get("cylinders", {})

        print("[SIG_DEBUG] algo:", payload.get("algo"))
        print("[SIG_DEBUG] topo:",
              f"V={topo.get('n_vertex')} E={topo.get('n_edge')} F={topo.get('n_face')} "
              f"W={topo.get('n_wire')} S={topo.get('n_solid')}")
        print("[SIG_DEBUG] props:",
              f"vol_q={props.get('vol_q')} area_q={props.get('area_q')} "
              f"moms_q={props.get('moms_q')} ratios_q={props.get('mom_ratio_q')} "
              f"ext={props.get('aabb_ext_q_sorted')} chi={props.get('chi')}")
        print("[SIG_DEBUG] frame_diag:", props.get("frame_diag"))

        print("[SIG_DEBUG] cylinders:",
              f"n_cyl_faces={cyl.get('n_cyl_faces')} desc_keys={len(cyl.get('desc_hist', {}))} "
              f"pos_keys={len(cyl.get('pos_bins', {}))}")

        # top radius bins
        r_hist = Counter({k: int(v) for k, v in (cyl.get("r_hist", {}) or {}).items()})
        if r_hist:
            print(f"[SIG_DEBUG] cylinders.r_hist top{_Q.DEBUG_TOP_N}:", _topn(r_hist, _Q.DEBUG_TOP_N))

        # top pos bins
        p_hist = Counter({k: int(v) for k, v in (cyl.get("pos_bins", {}) or {}).items()})
        if p_hist:
            print(f"[SIG_DEBUG] cylinders.pos_bins top{_Q.DEBUG_TOP_N}:", _topn(p_hist, _Q.DEBUG_TOP_N))

    except Exception as e:
        print("[SIG_DEBUG] (failed):", repr(e))


# ---------------------------------------------------------------------------
# Bounded diff helper
# ---------------------------------------------------------------------------

def diff_sig_features(shape_a, shape_b, *, mirror_invariant: bool = False, max_items: int = 20) -> Dict[str, Any]:
    """
    Compute payloads (sig or sig_free style depending on mirror_invariant),
    and return a bounded diff report.

    max_items caps per-histogram key diffs.
    """
    pa = _compute_feature_payload(shape_a, mirror_invariant=mirror_invariant)
    pb = _compute_feature_payload(shape_b, mirror_invariant=mirror_invariant)

    if "abort" in pa or "abort" in pb:
        return {"abort_a": pa.get("abort"), "abort_b": pb.get("abort")}

    out: Dict[str, Any] = {
        "algo_a": pa.get("algo"),
        "algo_b": pb.get("algo"),
        "groups_differ": [],
        "props_diff": {},
        "hist_diffs": {},
    }

    # Props scalar diffs
    props_a = pa.get("props", {})
    props_b = pb.get("props", {})
    for k in ("vol_q", "area_q", "moms_q", "mom_ratio_q", "aabb_ext_q_sorted", "chi", "frame_diag"):
        if props_a.get(k) != props_b.get(k):
            out["props_diff"][k] = {"a": props_a.get(k), "b": props_b.get(k)}

    def diff_hist(path: str, da: Dict[str, int], db: Dict[str, int]):
        keys = sorted(set(da.keys()) | set(db.keys()))
        diffs = []
        for k in keys:
            va, vb = int(da.get(k, 0)), int(db.get(k, 0))
            if va != vb:
                diffs.append((k, va, vb))
                if len(diffs) >= int(max_items):
                    break
        if diffs:
            out["hist_diffs"][path] = diffs

    # Compare groups
    for group in ("faces", "edges", "cylinders"):
        ga = pa.get(group, {}) or {}
        gb = pb.get(group, {}) or {}
        if ga != gb:
            out["groups_differ"].append(group)

    # faces
    fa, fb = pa.get("faces", {}) or {}, pb.get("faces", {}) or {}
    for h in ("type_counts", "area_q_hist", "area_frac_hist", "norm_hist", "cyl_r_hist",
              "cone_ang_hist", "sphere_r_hist", "torus_rr_hist", "plane_inner_wires_hist"):
        diff_hist(f"faces.{h}", fa.get(h, {}) or {}, fb.get(h, {}) or {})

    # edges
    ea, eb = pa.get("edges", {}) or {}, pb.get("edges", {}) or {}
    for h in ("type_counts", "len_q_hist", "circle_r_hist"):
        diff_hist(f"edges.{h}", ea.get(h, {}) or {}, eb.get(h, {}) or {})

    # cylinders
    ca, cb = pa.get("cylinders", {}) or {}, pb.get("cylinders", {}) or {}
    for h in ("r_hist", "desc_hist", "pos_bins"):
        diff_hist(f"cylinders.{h}", ca.get(h, {}) or {}, cb.get(h, {}) or {})

    return out

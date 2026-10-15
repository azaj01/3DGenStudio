"""Unroll bands of revolution into exact rectangles: treads, pipes, cans, rims.

A cylinder cut open flattens to a perfect rectangle with any solver. A tyre
tread does not: its profile bulges, the solver keeps lengths honest, and the
strip comes out as a curved "banana" that wastes atlas space and breaks a
tiling tread texture. Laid out analytically instead --

    u = mean radius x angle around the axis
    v = arc length along the profile

-- the band is an exact rectangle, at the cost of a little stretch where the
profile's radius departs from the mean.

An island counts as a band when it wraps (almost) all the way round an axis,
its surface faces outwards or inwards from that axis (|n . radial| high), its
reach is the same in every direction round the axis, and its profile is a
function of axial position (so arc length is well defined). The axis is
recovered from the geometry: on a surface of revolution every (p - c) x n is
perpendicular to the axis, so the axis is the null direction of those vectors;
the principal axes of the points are tried as well.

The analytic layout replaces the solver's only when its shear (95th
percentile) stays within ``slack`` of the solver's and it fills its bounding
box better.
"""
from __future__ import annotations

import numpy as np

from . import metrics as _metrics


def _axis_candidates(P, N):
    c = P.mean(axis=0)
    out = []
    C = np.cross(P - c, N)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    out.append(vt[-1])
    X = P - c
    _, vecs = np.linalg.eigh(X.T @ X)
    out.extend(vecs.T)
    return c, out


def _fill(uv, lf):
    a = np.abs(_metrics.signed_uv_area(uv, lf)).sum()
    ext = np.ptp(uv, axis=0)
    return float(a / max(ext[0] * ext[1], 1e-18))


def unroll_band(verts, uv, uniq, lf, min_wrap_deg=300.0, radial=0.8, reach_tol=0.06,
                slack=1.15, sectors=16):
    """Analytic rectangle layout for a band of revolution, or None."""
    P = verts[uniq]
    if len(lf) < 16:
        return None
    v = P[lf]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fa = np.linalg.norm(fn, axis=1)
    ok = fa > 0
    if ok.sum() < 16:
        return None
    fn = fn / np.where(ok, fa, 1.0)[:, None]
    fc = v.mean(axis=1)
    c, axes = _axis_candidates(fc[ok], fn[ok])
    best = None
    for ax in axes:
        a = ax / np.linalg.norm(ax)
        X = P - c
        z = X @ a
        R = X - np.outer(z, a)
        rho = np.linalg.norm(R, axis=1)
        if rho.mean() <= 1e-12:
            continue
        # faces must face away from / toward the axis
        rf = fc - c
        rf = rf - np.outer(rf @ a, a)
        rn = np.linalg.norm(rf, axis=1)
        good = ok & (rn > 0)
        if good.sum() < 0.9 * len(lf):
            continue
        dots = np.abs(np.einsum("ij,ij->i", fn[good], rf[good] / rn[good][:, None]))
        if np.average(dots, weights=fa[good]) < radial:
            continue
        # angle round the axis, and how much of the circle the band covers
        e1 = R[np.argmax(rho)] / max(rho.max(), 1e-18)
        e2 = np.cross(a, e1)
        th = np.arctan2(R @ e2, R @ e1)
        srt = np.sort(th)
        gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
        wrap = 2 * np.pi - gaps.max()
        if np.degrees(wrap) < min_wrap_deg:
            continue
        bins = ((th + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        reach = np.array([rho[bins == b].max() if (bins == b).any() else np.nan
                          for b in range(sectors)])
        reach = reach[np.isfinite(reach)]
        if len(reach) < sectors - 1 or np.ptp(reach) > reach_tol * reach.mean():
            continue
        best = (a, z, rho, th)
        break
    if best is None:
        return None
    a, z, rho, th = best

    # continuous angle across the island (it is a disk: the band is slit once)
    n = len(P)
    nbr = [[] for _ in range(n)]
    for f in lf:
        for i in range(3):
            nbr[f[i]].append(f[(i + 1) % 3])
            nbr[f[(i + 1) % 3]].append(f[i])
    cont = np.full(n, np.nan)
    cont[0] = th[0]
    stack = [0]
    while stack:
        u = stack.pop()
        for w in nbr[u]:
            if np.isnan(cont[w]):
                d = (th[w] - th[u] + np.pi) % (2 * np.pi) - np.pi
                cont[w] = cont[u] + d
                stack.append(w)
    if np.isnan(cont).any():
        return None

    # profile arc length as a function of axial position
    order = np.argsort(z)
    zs, rs = z[order], rho[order]
    q = max(np.ptp(z), np.ptp(rho)) * 0.002 or 1e-9
    keys = np.round(zs / q)
    uk, first = np.unique(keys, return_index=True)
    pz = np.array([zs[keys == k].mean() for k in uk])
    pr = np.array([rs[keys == k].mean() for k in uk])
    seg = np.hypot(np.diff(pz), np.diff(pr))
    arc = np.r_[0.0, np.cumsum(seg)]
    if len(pz) < 2:
        vv = z - z.min()
    else:
        vv = np.interp(z, pz, arc)
    uu = rho.mean() * cont
    new = np.column_stack([uu, vv])
    # keep the solver's orientation (u, v) vs (v, u) by winding
    if np.sign(_metrics.signed_uv_area(new, lf)).sum() < 0:
        new[:, 0] = -new[:, 0]
    if np.any(_metrics.signed_uv_area(new, lf) <= 0):
        return None

    old_p95 = _metrics.stretch_p95(P, lf, uv)
    new_p95 = _metrics.stretch_p95(P, lf, new)
    if new_p95 > slack * old_p95 + 0.02:
        return None
    if _fill(new, lf) <= _fill(uv, lf) + 0.02:
        return None
    return new


def unroll_bands(mesh, work):
    """Replace the layout of every band of revolution in ``work``. Returns ``(work, n)``."""
    out = []
    n = 0
    for fids, (uv, uniq, lf, info) in work:
        new = unroll_band(mesh.vertices, uv, uniq, lf)
        if new is None:
            out.append((fids, (uv, uniq, lf, info)))
        else:
            info = dict(info, method="unroll")
            out.append((fids, (new, uniq, lf, info)))
            n += 1
    return out, n

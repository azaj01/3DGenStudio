"""Lay thin strip islands back onto the island they came off.

A chamfer, a fillet, the rim of a cuff: a narrow band of faces whose normals
turn just enough that segmentation gave it its own chart. Flattened alone it is
a long thin ribbon -- the hardest shape to pack, with a seam down both sides.
Usually it can simply be laid back against one neighbour: its layout is already
fine, it is only *placed* wrong.

So no re-flattening happens here. For each ribbon (UV PCA short axis at most
``narrow`` x the long one):

1. the host is the neighbour sharing the longest 3D border with it;
2. only the longest unbroken run of that border is used -- pinning both ends
   of a strip that wraps a corner would fold its first row of faces;
3. a 2D similarity (rotation + uniform scale + shift, closed form) carries the
   strip's run vertices onto the host's;
4. the run vertices are snapped exactly onto the host's, which is what removes
   the seam there.

The move is refused when the fit is poor (residual over ``fit_tol`` x the run's
span, unless the strip is a hair-thin sliver), when the strip would land at a
texel density far from the host's, when any of its faces would flip, or when
the merged island overlaps itself.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import metrics as _metrics
from . import param as _param


def _shape_ratio(uv):
    c = uv - uv.mean(axis=0)
    ev = np.linalg.eigvalsh(c.T @ c)
    if ev[1] <= 0:
        return 0.0
    return float(np.sqrt(max(ev[0], 0.0) / ev[1]))


def _density(verts, uv, uniq, lf):
    _, a3 = _metrics.triangle_frames(verts[uniq], lf)
    a2 = np.abs(_metrics.signed_uv_area(uv, lf))
    return np.sqrt(a2.sum() / max(a3.sum(), 1e-18))


def _corner_local(lf_row, face_verts, v):
    """Local vertex id of global vertex ``v`` in one face (rows are aligned)."""
    k = int(np.nonzero(face_verts == v)[0][0])
    return int(lf_row[k])


def absorb_ribbons(mesh, work, narrow=0.15, hair=0.06, fit_tol=0.2, density_tol=1.5,
                   rounds=4, keep_edges=None):
    """Merge ribbon islands of ``work`` into their hosts. Returns ``(work, stats)``.

    ``work`` holds ``(fids, (uv, uniq, lf, info))`` per island, ``lf`` rows
    aligned with ``fids``. ``keep_edges`` is an optional set of welded vertex
    pairs ``(lo, hi)`` that must stay seams (a symmetry mid-line).
    """
    verts, faces = mesh.vertices, mesh.faces
    adj = mesh.adjacency
    ev = mesh.adjacency_edges
    elen = np.linalg.norm(verts[ev[:, 1]] - verts[ev[:, 0]], axis=1)
    work = list(work)
    stats = {"absorbed": 0, "candidates": 0}

    for _ in range(rounds):
        F = mesh.n_faces
        isl = np.full(F, -1, dtype=np.int64)
        row = np.full(F, -1, dtype=np.int64)
        for i, (fids, _) in enumerate(work):
            isl[fids] = i
            row[fids] = np.arange(len(fids))
        ia, ib = isl[adj[:, 0]], isl[adj[:, 1]]
        cross = np.nonzero((ia != ib) & (ia >= 0) & (ib >= 0))[0]

        strips = []
        for i, (fids, (uv, uniq, lf, info)) in enumerate(work):
            if len(fids) < 2:
                strips.append((float(mesh.face_areas[fids].sum()), i, 0.0))
                continue
            r = _shape_ratio(uv)
            if r <= narrow:
                strips.append((float(mesh.face_areas[fids].sum()), i, r))
        strips.sort()
        stats["candidates"] += len(strips)
        used = set()
        merged = 0
        replace = {}
        for _, s, ratio in strips:
            if s in used:
                continue
            on_s = cross[(ia[cross] == s) | (ib[cross] == s)]
            if len(on_s) == 0:
                continue
            other = np.where(ia[on_s] == s, ib[on_s], ia[on_s])
            hosts, inv = np.unique(other, return_inverse=True)
            blen = np.bincount(np.asarray(inv).ravel(), weights=elen[on_s])
            for hk in np.argsort(-blen):
                h = int(hosts[hk])
                if h in used:
                    continue
                edges = on_s[other == h]
                if keep_edges:
                    lo = np.minimum(ev[edges, 0], ev[edges, 1])
                    hi = np.maximum(ev[edges, 0], ev[edges, 1])
                    if any((int(a), int(b)) in keep_edges for a, b in zip(lo, hi)):
                        continue
                res = _try_absorb(mesh, work, s, h, edges, isl, row, elen,
                                  ratio <= hair, fit_tol, density_tol)
                if res is not None:
                    used.update((s, h))
                    replace[h] = res
                    replace[s] = None
                    merged += 1
                break                       # only the longest-border host is tried
        if merged == 0:
            break
        work = [replace[i] if i in replace else w for i, w in enumerate(work)]
        work = [w for w in work if w is not None]
        stats["absorbed"] += merged
    return work, stats


def _try_absorb(mesh, work, s, h, edges, isl, row, elen, is_hair, fit_tol, density_tol):
    verts, faces = mesh.vertices, mesh.faces
    adj = mesh.adjacency
    ev = mesh.adjacency_edges
    fs_, (uv_s, uniq_s, lf_s, info_s) = work[s]
    fh_, (uv_h, uniq_h, lf_h, info_h) = work[h]

    # vertex correspondences along every shared edge, in local ids
    pairs = {}                 # strip local -> host local
    run_edges = []             # (strip local a, strip local b, length)
    for e in edges:
        fa, fb = int(adj[e, 0]), int(adj[e, 1])
        f_s, f_h = (fa, fb) if isl[fa] == s else (fb, fa)
        ls = []
        for v in ev[e]:
            sl = _corner_local(lf_s[row[f_s]], faces[f_s], v)
            hl = _corner_local(lf_h[row[f_h]], faces[f_h], v)
            pairs.setdefault(sl, hl)
            ls.append(sl)
        run_edges.append((ls[0], ls[1], float(elen[e])))
    if not run_edges:
        return None

    # the longest connected run of the shared border
    ids = sorted(pairs)
    pos = {v: i for i, v in enumerate(ids)}
    ra = np.array([pos[a] for a, _, _ in run_edges])
    rb = np.array([pos[b] for _, b, _ in run_edges])
    n = len(ids)
    _, comp = connected_components(coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(n, n)),
                                   directed=False)
    clen = np.bincount(comp[ra], weights=[ln for _, _, ln in run_edges], minlength=comp.max() + 1)
    best = int(np.argmax(clen))
    run = [ids[i] for i in range(n) if comp[i] == best]
    if len(run) < 2:
        return None
    sl = np.array(run)
    hl = np.array([pairs[v] for v in run])

    # closed-form 2D similarity: z -> a z + t, carrying strip run onto host run
    P = uv_s[sl, 0] + 1j * uv_s[sl, 1]
    Q = uv_h[hl, 0] + 1j * uv_h[hl, 1]
    pc, qc = P.mean(), Q.mean()
    zp, zq = P - pc, Q - qc
    den = float(np.sum(np.abs(zp) ** 2))
    if den <= 1e-24:
        return None
    a = np.sum(np.conj(zp) * zq) / den
    resid = float(np.sqrt(np.mean(np.abs(a * zp - zq) ** 2)))
    span = float(np.abs(zq).max() * 2.0)
    if span <= 1e-18 or (not is_hair and resid > fit_tol * span):
        return None

    # texel density of the strip once moved, against the host's
    d_h = _density(verts, uv_h, uniq_h, lf_h)
    d_s = _density(verts, uv_s, uniq_s, lf_s) * abs(a)
    if d_h <= 0 or not (1.0 / density_tol <= d_s / d_h <= density_tol):
        return None

    Z = uv_s[:, 0] + 1j * uv_s[:, 1]
    Z = a * (Z - pc) + qc
    moved = np.column_stack([Z.real, Z.imag])
    moved[sl] = uv_h[hl]                                   # close the seam exactly
    before = np.sign(_metrics.signed_uv_area(uv_s, lf_s))
    after = _metrics.signed_uv_area(moved, lf_s)
    if np.any(np.sign(after) != before) or np.any(after == 0):
        return None

    # merged island: snapped strip vertices become the host's
    remap = np.full(len(uv_s), -1, dtype=np.int64)
    remap[sl] = hl
    rest = np.nonzero(remap < 0)[0]
    remap[rest] = len(uv_h) + np.arange(len(rest))
    uv_m = np.vstack([uv_h, moved[rest]])
    uniq_m = np.concatenate([uniq_h, uniq_s[rest]])
    lf_m = np.vstack([lf_h, remap[lf_s]])
    fids_m = np.concatenate([fh_, fs_])
    ov = _metrics.island_overlap(uv_m, lf_m, resolution=256)
    if ov > max(_param.OVERLAP_TOL, info_h["overlap"]):
        return None

    a_h = float(mesh.face_areas[fh_].sum())
    a_s = float(mesh.face_areas[fs_].sum())
    w = a_h + a_s or 1.0
    info = dict(info_h)
    info.update(
        flips=int(info_h["flips"] + info_s["flips"]),
        overlap=float(ov),
        angle_d=float((info_h["angle_d"] * a_h + info_s["angle_d"] * a_s) / w),
        area_d=float((info_h["area_d"] * a_h + info_s["area_d"] * a_s) / w),
        cut_edges=int(info_h.get("cut_edges", 0) + info_s.get("cut_edges", 0)),
    )
    return (fids_m, (uv_m, uniq_m, lf_m, info))

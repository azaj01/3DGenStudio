"""Open the creases of an island that came out badly stretched.

Some charts pass every test on the way in and still flatten with a lot of
shear: a stepped profile (a plinth, a moulding, a cuff with a ridge) has real
curvature but no clean place to split it in two. What relaxes it is a slit along
each sharp fold, so the flattener can open the profile out like a paper model.

For an island whose shear (area-weighted 95th percentile of sigma_max /
sigma_min) is over ``limit``, the interior edges of its chart with a dihedral
angle of at least ``angle`` are cut -- trying the fewest cuts first (80, then
50, then 30 degrees). Isolated one-edge nicks are dropped first; they open
nothing. The re-flattened island is kept only if its shear falls to at most
``gain`` x the old value and under ``limit``, with no new flips or folds.

A smooth chart has no creases to open. Then, if a ``split`` function is
given, the island is cut in two instead (the same crease-seeded two-region
split the fold repair uses) and kept split when the share of its area sheared
past 1.5x drops by at least ``split_gain``. That is the stretch-driven split of
the add-on's organic engine; any halves that turn out fine together are
offered back by the re-join pass later anyway.
"""
from __future__ import annotations

import numpy as np

from . import metrics as _metrics
from . import param as _param


def crease_edges(mesh, fids, angle_deg, prune_passes=4):
    """(K, 2) vertex pairs: interior chart edges at least ``angle_deg`` sharp."""
    inside = np.zeros(mesh.n_faces, dtype=bool)
    inside[fids] = True
    adj = mesh.adjacency
    keep = inside[adj[:, 0]] & inside[adj[:, 1]] & (mesh.adjacency_angle >= np.radians(angle_deg))
    e = np.sort(mesh.adjacency_edges[keep], axis=1)
    if len(e) == 0:
        return e
    e = np.unique(e, axis=0)
    for _ in range(prune_passes):
        deg = np.bincount(e.ravel(), minlength=int(e.max()) + 1)
        stub = (deg[e[:, 0]] <= 1) & (deg[e[:, 1]] <= 1)
        if not stub.any():
            break
        e = e[~stub]
        if len(e) == 0:
            break
    return e


def _bad_area(verts, res, fids, mesh):
    uv, uniq, lf, _ = res
    return _metrics.bad_share(verts[uniq], lf, uv) * float(mesh.face_areas[fids].sum())


def relax_stretched(mesh, work, flatten, split=None, limit=2.0, gain=0.9, min_faces=8,
                    angles=(80.0, 50.0, 30.0), split_gain=0.3, split_min_faces=24, depth=2):
    """Slit (or split) over-stretched islands in ``work``. Returns ``(work, stats)``.

    ``flatten(fids, cut_pairs)`` must return ``(uv, uniq, lf, info)``;
    ``split(fids, res)`` returns a list of face-id arrays.
    """
    verts = mesh.vertices
    stats = {"checked": 0, "relaxed": 0, "slit_edges": 0, "split": 0}
    out = []
    for fids, res in work:
        uv, uniq, lf, info = res
        if len(fids) < min_faces:
            out.append((fids, res))
            continue
        old = _metrics.stretch_p95(verts[uniq], lf, uv)
        if old <= limit:
            out.append((fids, res))
            continue
        stats["checked"] += 1
        area = float(mesh.face_areas[fids].sum())
        best = None
        for ang in angles:
            cuts = crease_edges(mesh, fids, ang)
            if len(cuts) == 0:
                continue
            clen = np.linalg.norm(verts[cuts[:, 1]] - verts[cuts[:, 0]], axis=1).sum()
            if clen < 0.02 * np.sqrt(area):
                continue
            r2 = flatten(fids, cuts)
            uv2, uniq2, lf2, info2 = r2
            if info2["flips"] > info["flips"]:
                continue
            if info2["overlap"] > max(_param.OVERLAP_TOL, info["overlap"]):
                continue
            new = _metrics.stretch_p95(verts[uniq2], lf2, uv2)
            if new <= min(gain * old, limit):
                best = (r2, len(cuts))
                break                         # fewest cuts that work
        if best is None and split is not None and len(fids) >= split_min_faces:
            parts = _split_relax(mesh, fids, res, flatten, split, split_gain,
                                 split_min_faces, depth, stats)
            if parts is not None:
                out.extend(parts)
                continue
        if best is None:
            out.append((fids, res))
        else:
            out.append((fids, best[0]))
            stats["relaxed"] += 1
            stats["slit_edges"] += best[1]
    return out, stats


def _split_relax(mesh, fids, res, flatten, split, split_gain, min_faces, depth, stats):
    """Split a smooth, stretched island in two while that lowers its shear."""
    verts = mesh.vertices
    info = res[3]
    before = _bad_area(verts, res, fids, mesh)
    if before <= 0:
        return None
    pieces = split(fids, res)
    if len(pieces) < 2 or len(pieces) > 4:
        return None
    parts = [(p, flatten(p)) for p in pieces]
    if sum(r[3]["flips"] for _, r in parts) > info["flips"]:
        return None
    if any(r[3]["overlap"] > max(_param.OVERLAP_TOL, info["overlap"]) for _, r in parts):
        return None
    after = sum(_bad_area(verts, r, p, mesh) for p, r in parts)
    if after > (1.0 - split_gain) * before:
        return None
    stats["split"] += 1
    out = []
    for p, r in parts:
        if depth > 1 and len(p) >= min_faces and                 _metrics.bad_share(verts[r[1]], r[2], r[0]) > 0.06:
            sub = _split_relax(mesh, p, r, flatten, split, split_gain, min_faces,
                               depth - 1, stats)
            if sub is not None:
                out.extend(sub)
                continue
        out.append((p, r))
    return out

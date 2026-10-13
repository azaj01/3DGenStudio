"""Give seams back: re-join neighbouring islands after they have been flattened.

Every stage before this one errs on the side of cutting. Segmentation stops a
chart when its normal cone fills, the merge pass refuses any union that folds,
and charts that still fold are split again. Each of those is right locally, but
together they leave more islands than the surface needs: two charts split
because one *candidate* union folded may well flatten cleanly once the
neighbourhood has settled.

So once every island has its layout, adjacent pairs are offered back, longest
shared border first (a long border is the most seam removed per join). The
union is re-flattened and kept only when it is as good as the pair it replaces:

* no flips and no self-overlap;
* shear (area-weighted 95th percentile of sigma_max / sigma_min) under a
  ceiling -- ``limit``, or the worse of the two islands x ``give`` when they
  were already above it. Very small or sliver-thin islands get the lenient
  ceiling, since a seam around a crumb costs more than a little shear;
* texel-density drift (share of the island more than 25% off its median
  scale) no worse than ``drift_limit`` or than the pair already had. This
  gate is ours, not the add-on's: shear is blind to it, and without it whole
  heads re-joined into conformal islands with 70% of the surface off-scale;
* its bounding box no larger than ``box_slack`` x the two boxes it replaces,
  so a join cannot trade a seam for a sprawling L that wastes the atlas.

A global guard watches the share of the whole surface sheared past 1.5x: a
round that pushes it above ``cap`` (or above where it started, if it started
higher) is undone and the pass stops.
"""
from __future__ import annotations

import time

import numpy as np

from . import metrics as _metrics
from .postprocess import align_island


def _density_drift(a3, a2):
    """Share of an island's area whose texel density is off its median by >25%."""
    ok = a3 > 1e-18
    if not ok.any():
        return 1.0
    s = np.sqrt(a2[ok] / a3[ok])
    w = a3[ok]
    o = np.argsort(s)
    med = s[o][min(np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5), len(s) - 1)]
    if med <= 0:
        return 1.0
    off = (s < 0.8 * med) | (s > 1.25 * med)
    return float(w[off].sum() / w.sum())


def _island_stats(verts, uv, uniq, lf):
    """(shear p95, 3D area, bbox area at uniform density, is_sliver, density drift)."""
    lv = verts[uniq]
    p95 = _metrics.stretch_p95(lv, lf, uv)
    _, a3 = _metrics.triangle_frames(lv, lf)
    area3 = float(a3.sum())
    a2 = np.abs(_metrics.signed_uv_area(uv, lf))
    ua = float(a2.sum())
    drift = _density_drift(a3, a2)
    if ua <= 1e-18 or area3 <= 1e-18:
        return p95, area3, 0.0, True, drift
    aligned = align_island(uv * np.sqrt(area3 / ua))
    ext = aligned.max(axis=0) - aligned.min(axis=0)
    c = uv - uv.mean(axis=0)
    ev = np.linalg.eigvalsh(c.T @ c)
    sliver = bool(ev[1] <= 0 or np.sqrt(max(ev[0], 0.0) / ev[1]) <= 0.15)
    return p95, area3, float(ext[0] * ext[1]), sliver, drift


def rejoin_islands(mesh, work, flatten, flatten_fast=None, limit=1.45, give=1.02, floor=1.45,
                   tiny=0.0015, box_slack=1.15, cap=0.09, drift_limit=0.10, rounds=8,
                   budget=30.0,
                   progress=None):
    """Re-join adjacent islands of ``work`` where the union flattens as well.

    ``work`` is a list of ``(fids, (uv, uniq, lf, info))`` -- one entry per
    island, as produced by the per-chart flattening. ``flatten(fids)`` must
    return the same ``(uv, uniq, lf, info)`` tuple for a face set.
    ``flatten_fast`` (same contract, cheaper -- LSCM alone) screens every
    candidate first; most joins are refused, so the full flatten then runs
    only on the few that pass. Returns ``(work, stats)``.
    """
    verts = mesh.vertices
    adj = mesh.adjacency
    ev = mesh.adjacency_edges
    elen = np.linalg.norm(verts[ev[:, 1]] - verts[ev[:, 0]], axis=1)
    total_area = float(mesh.face_areas.sum()) or 1.0
    t0 = time.time()
    work = list(work)
    stats = {"joined": 0, "tried": 0, "rounds": 0, "reverted_rounds": 0}

    island = [_island_stats(verts, r[0], r[1], r[2]) for _, r in work]
    # A refused pair stays refused while neither island changes, and most of
    # every round's candidates are pairs an earlier round already refused.
    refused = set()

    def ikey(fids):
        return hash(fids.tobytes())

    def global_bad():
        num = 0.0
        for (_, r), st in zip(work, island):
            lv = verts[r[1]]
            num += _metrics.bad_share(lv, r[2], r[0]) * st[1]
        return num / total_area

    start_bad = global_bad()
    line = max(cap, start_bad)
    stats["shear_share_before"] = start_bad

    for rnd in range(rounds):
        if time.time() - t0 > budget:
            break
        face_isl = np.full(mesh.n_faces, -1, dtype=np.int64)
        for i, (fids, _) in enumerate(work):
            face_isl[fids] = i
        ia, ib = face_isl[adj[:, 0]], face_isl[adj[:, 1]]
        cross = (ia != ib) & (ia >= 0) & (ib >= 0)
        if not cross.any():
            break
        lo, hi = np.minimum(ia[cross], ib[cross]), np.maximum(ia[cross], ib[cross])
        n = len(work)
        key, inv = np.unique(lo * n + hi, return_inverse=True)
        border = np.bincount(np.asarray(inv).ravel(), weights=elen[cross])
        order = np.argsort(-border)

        snapshot = (list(work), list(island))
        used = set()
        new_work = []
        new_island = []
        joined_round = 0
        for k in order:
            if time.time() - t0 > budget:
                break
            a, b = int(key[k] // n), int(key[k] % n)
            if a in used or b in used:
                continue
            fa_, ra = work[a]
            fb_, rb = work[b]
            sa, sb = island[a], island[b]
            pk = (ikey(fa_), ikey(fb_))
            if pk in refused:
                continue
            stats["tried"] += 1
            union = np.sort(np.concatenate([fa_, fb_]))
            small = (min(sa[1], sb[1]) < tiny * total_area) or sa[3] or sb[3]
            crumb = min(sa[1], sb[1]) < tiny * total_area
            was = max(sa[0], sb[0])
            if small:
                ceiling = max(limit, was * give)
            else:
                ceiling = max(min(limit, max(was * give, floor)), was)

            def judge(res):
                """(island stats, None) if the union is acceptable, else (None, reason)."""
                uv, uniq, lf, info = res
                if info["flips"] > ra[3]["flips"] + rb[3]["flips"]:
                    return None, "flips"
                if info["overlap"] > max(0.002, ra[3]["overlap"], rb[3]["overlap"]):
                    return None, "overlap"
                if info.get("components", 1) != 1:
                    return None, "torn"
                su = _island_stats(verts, uv, uniq, lf)
                if su[0] > ceiling:
                    return None, "shear"
                # Shear alone cannot see a conformal island whose texel
                # density drifts: a whole head flattens with almost no shear
                # and half its surface at the wrong scale.
                if su[4] > max(drift_limit, max(sa[4], sb[4]) * give + 0.01):
                    return None, "drift"
                # Only true crumbs skip the box test; a sliver joined onto a
                # panel is exactly how an atlas fills up with L-shapes.
                if not crumb and su[2] > box_slack * (sa[2] + sb[2]):
                    return None, "box"
                return su, None

            # Screen with LSCM alone. It is the least-shear flattening there
            # is, so a union it cannot get under the shear ceiling is refused
            # outright, as is one that tears or sprawls. Folds and texel-density
            # drift are another matter: those are what the planar and ARAP
            # candidates of the full search are for, so they get it.
            if flatten_fast is not None:
                fres = flatten_fast(union)
                _, why = judge(fres)
                if why in ("shear", "torn", "box"):
                    refused.add(pk)
                    continue
            res = flatten(union)
            su, why = judge(res)
            if su is None:
                refused.add(pk)
                continue
            used.update((a, b))
            new_work.append((union, res))
            new_island.append(su)
            joined_round += 1
        if joined_round == 0:
            break
        keep = [i for i in range(len(work)) if i not in used]
        work = [work[i] for i in keep] + new_work
        island = [island[i] for i in keep] + new_island
        bad = global_bad()
        stats["rounds"] += 1
        if bad > line:
            work, island = snapshot
            stats["reverted_rounds"] += 1
            break
        stats["joined"] += joined_round
        if progress is not None:
            progress((rnd + 1) / rounds)
    stats["shear_share_after"] = global_bad()
    return work, stats

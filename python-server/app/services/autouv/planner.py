"""Cut a character the way an artist would: limbs at the joints, front from back.

Normal-cone segmentation knows nothing about anatomy. On a body it grows charts
until the curvature budget runs out, which leaves borders wherever that
happened to be -- across a forearm, through a cheek -- and a long tail of small
charts. An artist cuts by rule instead: each limb off at its narrowest point
(wrist, ankle, neck), and the trunk front from back along the silhouette.

This module produces those labels from geometry alone:

1. **Limbs** -- geodesic farthest-point sampling from the vertex nearest the
   centre picks the extremities (hands, feet, head, tail, ears), up to ``k`` of
   them and none closer than ``floor`` x the bounding-box diagonal. Every
   vertex joins its geodesically nearest extremity, or the centre (the trunk).
2. **Rings** -- inside each limb, the cross-section length is measured at
   ``levels`` geodesic distances from its tip. Clear local minima (a dip of at
   least ``prom`` of the profile's range) are the joints; up to ``per_limb``
   per limb, kept at least 6 levels apart.
3. **Front / back** -- the face normals' component along the model's thinnest
   axis (depth, for a standing figure), smoothed over the surface; its sign
   is the side. The silhouette becomes the cut.
4. Pieces under ``min_part`` faces are merged into the neighbour they share
   the longest border with.

The result is only a proposal: the unwrapper lays it out with the same passes
as its own segmentation and keeps whichever comes out better.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def _edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def _merge_small(mesh, labels, min_part, rounds=10):
    """Fold pieces under ``min_part`` faces into the longest-bordering neighbour."""
    adj = mesh.adjacency
    ev = mesh.adjacency_edges
    elen = np.linalg.norm(mesh.vertices[ev[:, 1]] - mesh.vertices[ev[:, 0]], axis=1)
    labels = labels.copy()
    for _ in range(rounds):
        sizes = np.bincount(labels)
        small = np.nonzero((sizes > 0) & (sizes < min_part))[0]
        if len(small) == 0:
            break
        la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
        diff = la != lb
        changed = False
        target = {}
        for s in small[np.argsort(sizes[small])]:
            on = diff & ((la == s) | (lb == s))
            if not on.any():
                continue
            other = np.where(la[on] == s, lb[on], la[on])
            nb, inv = np.unique(other, return_inverse=True)
            w = np.bincount(np.asarray(inv).ravel(), weights=elen[on])
            target[int(s)] = int(nb[np.argmax(w)])
        for s, t in target.items():
            for _ in range(32):                     # follow chains of merges
                if t not in target or target[t] == s:
                    break
                t = target[t]
            labels[labels == s] = t
            changed = True
        if not changed:
            break
    _, labels = np.unique(labels, return_inverse=True)
    return np.asarray(labels).ravel()


def character_labels(mesh, k=12, floor=0.10, levels=60, prom=0.12, per_limb=3,
                     min_limb_faces=24, min_part=30, smooth_rounds=6):
    """Chart labels for ``mesh`` cut at limbs, joints and the silhouette."""
    V = mesh.vertices
    F = mesh.faces
    nV = len(V)
    E = _edges(F)
    L = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
    G = coo_matrix((L, (E[:, 0], E[:, 1])), shape=(nV, nV)).tocsr()
    G = (G + G.T).tocsr()
    diag = float(np.linalg.norm(np.ptp(V, axis=0))) or 1.0

    used = np.zeros(nV, dtype=bool)
    used[F.ravel()] = True
    n_comp, vcomp = connected_components(G, directed=False)
    region = np.full(nV, -1, dtype=np.int64)
    tip_of_region = {}                 # region id -> distance field from its tip
    nxt = 0
    for c in range(n_comp):
        cv = np.nonzero((vcomp == c) & used)[0]
        if len(cv) == 0:
            continue
        if len(cv) < 3:
            region[cv] = nxt
            nxt += 1
            continue
        centre = cv[np.argmin(np.linalg.norm(V[cv] - V[cv].mean(axis=0), axis=1))]
        d_centre = dijkstra(G, indices=centre)
        seeds = [d_centre]
        dmin = d_centre.copy()
        tips = []
        while len(tips) < k:
            cand = cv[np.argmax(dmin[cv])]
            if not np.isfinite(dmin[cand]) or dmin[cand] < floor * diag:
                break
            d = dijkstra(G, indices=cand)
            tips.append(d)
            seeds.append(d)
            dmin = np.minimum(dmin, d)
        D = np.stack([s[cv] for s in seeds])          # (seeds, verts of comp)
        near = np.argmin(D, axis=0)
        region[cv] = nxt + near
        for i, d in enumerate(tips):
            tip_of_region[nxt + 1 + i] = d
        nxt += len(seeds)

    # face region = majority of its corners (ties go to the first corner)
    fr = region[F]
    face_region = np.where((fr[:, 1] == fr[:, 2]) & (fr[:, 1] != fr[:, 0]), fr[:, 1], fr[:, 0])

    # ---- ring cuts at the narrowest cross-sections of every limb ------------
    seg = np.zeros(len(F), dtype=np.int64)
    for r, d in tip_of_region.items():
        fmask = face_region == r
        if fmask.sum() < min_limb_faces:
            continue
        rv = np.unique(F[fmask])
        lo, hi = float(d[rv].min()), float(d[rv].max())
        if hi - lo < 0.02 * diag:
            continue
        inside = np.zeros(nV, dtype=bool)
        inside[rv] = True
        em = inside[E[:, 0]] & inside[E[:, 1]]
        ea, eb, el = d[E[em, 0]], d[E[em, 1]], L[em]
        xs = lo + (np.arange(levels) + 0.5) / levels * (hi - lo)
        per = np.array([el[(ea - x) * (eb - x) < 0].sum() for x in xs])
        rng = float(per.max() - per.min())
        if rng <= 0:
            continue
        cands = []
        for i in range(1, levels - 1):
            if not (per[i] <= per[i - 1] and per[i] <= per[i + 1]):
                continue
            left = per[max(0, i - 12):i].max()
            right = per[i + 1:i + 13].max()
            p = (min(left, right) - per[i]) / rng
            if p > prom:
                cands.append((p, i))
        cands.sort(reverse=True)
        chosen = []
        for p, i in cands:
            if all(abs(i - j) >= 6 for j in chosen):
                chosen.append(i)
            if len(chosen) >= per_limb:
                break
        if not chosen:
            continue
        cuts = np.sort(xs[chosen])
        fd = d[F[fmask]].mean(axis=1)
        seg[fmask] = np.searchsorted(cuts, fd) + 1

    # ---- front / back along the thinnest axis, smoothed over the surface ----
    axis = int(np.argmin(np.ptp(V, axis=0)))
    val = mesh.face_normals[:, axis].copy()
    adj = mesh.adjacency
    ev = mesh.adjacency_edges
    w = np.linalg.norm(V[ev[:, 1]] - V[ev[:, 0]], axis=1)
    nF = len(F)
    W = coo_matrix((np.r_[w, w], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                   shape=(nF, nF)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    for _ in range(smooth_rounds):
        val = 0.5 * val + 0.5 * (W @ val) / deg
    side = (val >= 0).astype(np.int64)

    # ---- pieces = connected runs of equal (limb, ring, side) -----------------
    key = (face_region * 8 + np.minimum(seg, 7)) * 2 + side
    same = key[adj[:, 0]] == key[adj[:, 1]]
    g = coo_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])), shape=(nF, nF))
    _, labels = connected_components(g, directed=False)
    return _merge_small(mesh, labels.astype(np.int64), min_part)

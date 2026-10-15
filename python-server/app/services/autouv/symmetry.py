"""A seam down the mirror plane of a symmetric model.

Artists cut characters, vehicles and props down their middle so the two halves
lay out as mirror images and texture work done on one side reads on the other.
This module finds the plane and turns it into a protected seam:

1. **Plane** -- for each candidate normal (the model's principal axes and the
   world axes) the vertices are reflected through a plane at their centroid
   and matched to their nearest vertex. The plane is moved to the mean
   midpoint of the matched pairs and the match repeated (3 rounds). The best
   candidate wins if at least ``min_share`` of the vertices found a mirror
   partner within ``tol`` x the model size.
2. **Mid-line** -- mesh edges whose two ends lie on the plane (within
   ``tol`` x size) and whose two faces sit on opposite sides of it. Only when
   there are at least 8 of them does the plane run along the mesh's own
   edges; otherwise a cut there would zig-zag through faces and is skipped.
3. **Split** -- every chart that straddles the plane is split by side. The
   mid-line edges are returned so later passes (island re-join, ribbon
   absorb) leave them cut.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def find_plane(vertices, tol=0.004, min_share=0.6, max_samples=20000, seed=0):
    """``(normal, offset, share)`` of the best mirror plane, or None."""
    V = np.asarray(vertices, dtype=np.float64)
    if len(V) < 8:
        return None
    size = float(np.ptp(V, axis=0).max()) or 1.0
    eps = tol * size
    rng = np.random.default_rng(seed)
    S = V if len(V) <= max_samples else V[rng.choice(len(V), max_samples, replace=False)]
    tree = cKDTree(V)
    c = V.mean(axis=0)
    X = V - c
    _, vecs = np.linalg.eigh(X.T @ X)
    best = None
    for n in list(vecs.T) + list(np.eye(3)):
        n = n / np.linalg.norm(n)
        d = float(c @ n)
        share = 0.0
        for _ in range(3):
            refl = S - 2.0 * ((S @ n) - d)[:, None] * n
            dist, j = tree.query(refl)
            hit = dist <= eps
            share = float(hit.mean())
            if hit.sum() < 8:
                break
            mid = 0.5 * (S[hit] + V[j[hit]])
            d = float(np.mean(mid @ n))
        if best is None or share > best[2]:
            best = (n, d, share)
    if best is None or best[2] < min_share:
        return None
    return best


def mid_line(mesh, plane, tol=0.004):
    """Boolean mask over ``mesh.adjacency`` rows: edges that run along the plane."""
    n, d, _ = plane
    V = mesh.vertices
    size = float(np.ptp(V, axis=0).max()) or 1.0
    eps = tol * size
    sd = V @ n - d
    ev = mesh.adjacency_edges
    on = (np.abs(sd[ev[:, 0]]) <= eps) & (np.abs(sd[ev[:, 1]]) <= eps)
    fc = V[mesh.faces].mean(axis=1) @ n - d
    adj = mesh.adjacency
    opposite = np.sign(fc[adj[:, 0]]) * np.sign(fc[adj[:, 1]]) < 0
    return on & opposite


def split_by_plane(mesh, labels, plane):
    """Split every chart that straddles the plane into its two sides."""
    n, d, _ = plane
    side = (mesh.vertices[mesh.faces].mean(axis=1) @ n - d >= 0).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    out = labels * 2 + side
    _, out = np.unique(out, return_inverse=True)
    return np.asarray(out).ravel()

"""Per-chart post-processing: straighten and normalise texel density.

* Align: rotate each island to its minimum-area bounding rectangle. Artists do
  this by hand because axis-aligned, roughly-rectangular islands pack tighter
  and make texture painting predictable.
* Texel density: scale every island by a single global factor so that
  ``uv_area / surface_area`` is identical across charts -- no island ends up
  blurrier than its neighbours.
"""
from __future__ import annotations

import numpy as np


def _rotate(X, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    return X @ np.array([[ca, -sa], [sa, ca]]).T


def align_island(uv):
    """Rotate an island to its smallest bounding rectangle; return a copy.

    The packer sees only bounding boxes, so the orientation that matters is the
    one with the least box area, not the principal axis. For long straight
    islands the two agree; for L-shapes, fans and the big irregular islands a
    re-join pass produces they do not, and the PCA box can be much larger. The
    minimum-area rectangle has one side flush with an edge of the convex hull
    (rotating calipers), so only those angles need testing.

    Rotation only: the island is then turned a further 90 degrees if it came
    out taller than wide. (This used to swap x and y, which *mirrors* the
    island -- half of all islands shipped with reversed winding, so painted
    text read backwards and tangent-space handedness flipped between islands.)
    """
    c = uv.mean(axis=0)
    X = uv - c
    if len(X) < 3:
        return uv - uv.min(axis=0)
    angle = None
    try:
        from scipy.spatial import ConvexHull
        H = X[ConvexHull(X).vertices]
        e = np.roll(H, -1, axis=0) - H
        cand = np.arctan2(e[:, 1], e[:, 0])
        best = None
        for t in cand:
            R = _rotate(H, -t)
            area = float(np.ptp(R[:, 0]) * np.ptp(R[:, 1]))
            if best is None or area < best[0] - 1e-15:
                best = (area, t)
        angle = best[1]
    except Exception:  # degenerate (collinear) island: fall back to PCA
        w, V = np.linalg.eigh(X.T @ X)
        axis = V[:, np.argmax(w)]
        angle = np.arctan2(axis[1], axis[0])
    out = _rotate(X, -angle)
    # keep islands "wide" (w >= h) for tidy shelf packing -- by rotating
    if np.ptp(out[:, 1]) > np.ptp(out[:, 0]):
        out = np.column_stack([-out[:, 1], out[:, 0]])
    return out - out.min(axis=0)


def uv_area(uv, faces):
    """True UV area = sum of triangle areas in parameter space."""
    Q = uv[faces]
    a = (Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1]) - \
        (Q[:, 2, 0] - Q[:, 0, 0]) * (Q[:, 1, 1] - Q[:, 0, 1])
    return float(np.abs(a).sum() * 0.5)


def normalize_texel_density(islands, faces_list, areas3d):
    """Scale every island so uv_area / surface_area is the same constant.

    The constant is irrelevant (packing rescales the whole atlas afterwards);
    what matters is that the *ratio* is uniform, i.e. identical texel density.
    """
    out = []
    for uv, faces, a3 in zip(islands, faces_list, areas3d):
        ua = uv_area(uv, faces)
        # want uv_area_new / a3 == 1  ->  scale^2 * ua / a3 == 1
        s = np.sqrt(max(a3, 1e-12) / max(ua, 1e-12))
        scaled = uv * s
        scaled -= scaled.min(axis=0)
        out.append(scaled)
    return out

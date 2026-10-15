"""Find repeated parts so every copy can share one master's UVs.

Kit-bashed and generated models repeat themselves: four identical wheels, a
row of bolts, the spokes of a rim, the blades of a fan. Unwrapping each copy
separately spends texture space on pixels that will be painted the same, so
the copies are taken out of the unwrap and laid exactly on their master's
islands afterwards -- the master gets several times the texel density for the
same atlas. It is opt-in: stacked copies cannot be painted differently.

Two kinds of repeat are found:

* **clones** -- separate solids that are rigid copies of one another. Solids
  are bucketed by face and vertex count and by their sorted
  centroid-to-vertex distances, then a rigid transform is fitted from three
  anchor points (farthest from the centre, farthest off that line, farthest
  off that plane) and checked vertex by vertex. Mirror images are not copies:
  their UVs would come out reversed.
* **turns** -- rotational repeats inside one solid (spokes welded into a rim).
  A solid is turned by 2*pi/n about its principal axes and the world axes;
  when at least ``turn_cover`` of its area lands on faces of the same area,
  faces are grouped into orbits that close in exactly n steps, and one
  contiguous wedge -- one face per orbit, grown from a single seed -- is kept
  as the master.

The result maps each copy face to ``(master face, corner permutation)``.
Nothing is stacked when the copies would cover under ``min_share`` of the
model: on a body, a pair of near-identical ears is not worth a shared island.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _kabsch(A, B):
    """Rotation R and translation t with R @ a + t ~= b (no reflection)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        return None, None
    return R, cb - R @ ca


def _anchors(P, c):
    d = np.linalg.norm(P - c, axis=1)
    a1 = int(np.argmax(d))
    u = P[a1] - c
    u /= max(np.linalg.norm(u), 1e-18)
    off = P - c - np.outer((P - c) @ u, u)
    a2 = int(np.argmax(np.linalg.norm(off, axis=1)))
    nrm = np.cross(u, P[a2] - c)
    nn = np.linalg.norm(nrm)
    if nn <= 1e-18:
        return [a1, a2]
    a3 = int(np.argmax(np.abs((P - c) @ (nrm / nn))))
    return [a1, a2, a3]


def _corner_perm(V, fm, fc, R, t):
    """For each corner of copy face ``fc``, the matching corner of ``fm``."""
    pm = V[fm] @ R.T + t                       # (n, 3, 3) master corners, moved
    pc = V[fc]
    d = np.linalg.norm(pc[:, :, None, :] - pm[:, None, :, :], axis=3)   # (n, 3, 3)
    return np.argmin(d, axis=2)


def _match_faces(V, F, fm_ids, fc_ids, R, t, eps):
    """Map faces of a moved master onto copy faces by centroid. None if any misses."""
    cm = V[F[fm_ids]].mean(axis=1) @ R.T + t
    cc = V[F[fc_ids]].mean(axis=1)
    d, j = cKDTree(cc).query(cm)
    if np.any(d > eps) or len(np.unique(j)) != len(j):
        return None
    return fc_ids[j]


def find_clones(mesh, tol=1e-3, max_tries=240):
    """Copy face -> (master face, perm) for solids that are rigid copies."""
    V, F = mesh.vertices, mesh.faces
    comp = mesh.components
    extent = float(np.ptp(V, axis=0).max()) or 1.0
    out = {}
    groups = {}
    for c in np.unique(comp):
        fids = np.nonzero(comp == c)[0]
        vids = np.unique(F[fids])
        groups.setdefault((len(fids), len(vids)), []).append((fids, vids))
    for members in groups.values():
        if len(members) < 2:
            continue
        sig = []
        for fids, vids in members:
            P = V[vids]
            c = P.mean(axis=0)
            sig.append((np.sort(np.linalg.norm(P - c, axis=1)), c))
        taken = set()
        for i, (fm, vm) in enumerate(members):
            if i in taken:
                continue
            Pm = V[vm]
            cm = sig[i][1]
            size = float(np.ptp(Pm, axis=0).max())
            eps = max(tol * size, 1e-5 * extent)
            anc = _anchors(Pm, cm)
            rm = np.linalg.norm(Pm[anc] - cm, axis=1)
            for j in range(i + 1, len(members)):
                if j in taken:
                    continue
                fc, vc = members[j]
                if np.abs(sig[i][0] - sig[j][0]).max() > eps:
                    continue
                Pc = V[vc]
                cc = sig[j][1]
                rc = np.linalg.norm(Pc - cc, axis=1)
                cands = [np.nonzero(np.abs(rc - r) <= eps)[0][:32] for r in rm]
                if any(len(x) == 0 for x in cands):
                    continue
                tree = cKDTree(Pc)
                found = None
                tries = 0
                for a in cands[0]:
                    for b in cands[1]:
                        if abs(np.linalg.norm(Pc[a] - Pc[b]) - np.linalg.norm(Pm[anc[0]] - Pm[anc[1]])) > eps:
                            continue
                        for c3 in (cands[2] if len(cands) > 2 else [None]):
                            tries += 1
                            if tries > max_tries:
                                break
                            idx = [a, b] + ([c3] if c3 is not None else [])
                            R, t = _kabsch(np.vstack([cm, Pm[anc[:len(idx)]]]),
                                           np.vstack([cc, Pc[idx]]))
                            if R is None:
                                continue
                            d, k = tree.query(Pm @ R.T + t)
                            if np.all(d <= eps) and len(np.unique(k)) == len(k):
                                found = (R, t)
                                break
                        if found or tries > max_tries:
                            break
                    if found or tries > max_tries:
                        break
                if found is None:
                    continue
                R, t = found
                copy = _match_faces(V, F, fm, fc, R, t, eps)
                if copy is None:
                    continue
                perm = _corner_perm(V, F[fm], F[copy], R, t)
                for mf, cf, p in zip(fm, copy, perm):
                    out[int(cf)] = (int(mf), p)
                taken.add(j)
    return out


def _rotation(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def find_turns(mesh, tol=1e-3, turn_cover=0.75, min_rings=16, max_n=12, min_faces=24):
    """Copy face -> (master face, perm) for rotational repeats within a solid."""
    V, F = mesh.vertices, mesh.faces
    comp = mesh.components
    areas = mesh.face_areas
    extent = float(np.ptp(V, axis=0).max()) or 1.0
    adj = mesh.adjacency
    out = {}
    for c in np.unique(comp):
        fids = np.nonzero(comp == c)[0]
        if len(fids) < min_faces:
            continue
        cen = V[F[fids]].mean(axis=1)
        a = areas[fids]
        if a.sum() <= 0:
            continue
        origin = (cen * a[:, None]).sum(axis=0) / a.sum()
        X = cen - origin
        _, vecs = np.linalg.eigh((X * a[:, None]).T @ X)
        axes = list(vecs.T) + list(np.eye(3))
        span = float(np.ptp(cen, axis=0).max())
        med = float(np.median(np.sqrt(a)))
        eps = min(max(tol * span, tol * extent), 0.3 * med)
        tree = cKDTree(cen)
        amax = float(a.max())
        best = None
        for axis in axes:
            for n in range(min(max_n, len(fids) // 4), 1, -1):
                R = _rotation(axis, 2 * np.pi / n)
                moved = (cen - origin) @ R.T + origin
                d, j = tree.query(moved)
                ok = (d <= eps) & (np.abs(a[j] - a) <= 1e-3 * amax)
                if a[ok].sum() < turn_cover * a.sum():
                    continue
                mp = np.where(ok, j, -1)
                # orbits that close in exactly n steps
                ring = np.full(len(fids), -1, dtype=np.int64)
                rings = 0
                for s in range(len(fids)):
                    if ring[s] >= 0 or mp[s] < 0:
                        continue
                    orbit = [s]
                    x = mp[s]
                    while x >= 0 and x != s and len(orbit) <= n:
                        orbit.append(x)
                        x = mp[x]
                    if x == s and len(orbit) == n and np.all(ring[orbit] < 0):
                        ring[orbit] = rings
                        rings += 1
                cover = a[ring >= 0].sum() / a.sum()
                if rings >= min_rings and cover >= turn_cover:
                    if best is None or n > best[0]:
                        best = (n, R, mp, ring, rings)
                    break
        if best is None:
            continue
        n, R, mp, ring, rings = best
        # one contiguous wedge: grow from a seed, claiming one face per orbit
        local = np.full(mesh.n_faces, -1, dtype=np.int64)
        local[fids] = np.arange(len(fids))
        inside = (local[adj[:, 0]] >= 0) & (local[adj[:, 1]] >= 0)
        la, lb = local[adj[inside, 0]], local[adj[inside, 1]]
        g = coo_matrix((np.ones(len(la)), (la, lb)), shape=(len(fids), len(fids))).tocsr()
        g = (g + g.T).tocsr()
        claimed = np.zeros(rings, dtype=bool)
        master = np.zeros(len(fids), dtype=bool)
        seed = int(np.nonzero(ring >= 0)[0][0])
        queue = [seed]
        claimed[ring[seed]] = True
        master[seed] = True
        while queue:
            f = queue.pop(0)
            for nb in g.indices[g.indptr[f]:g.indptr[f + 1]]:
                r = ring[nb]
                if r >= 0 and not claimed[r]:
                    claimed[r] = True
                    master[nb] = True
                    queue.append(int(nb))
        for m in np.nonzero(master)[0]:
            x = m
            Rk = np.eye(3)
            for _ in range(n - 1):
                x = mp[x]
                Rk = R @ Rk
                tk = origin - Rk @ origin
                perm = _corner_perm(V, F[[fids[m]]], F[[fids[x]]], Rk, tk)[0]
                out[int(fids[x])] = (int(fids[m]), perm)
    return out


def find_repeats(mesh, min_share=0.02):
    """All repeats: copy face -> (master face, corner permutation)."""
    rep = find_clones(mesh)
    turns = find_turns(mesh)
    for f, v in turns.items():
        if f not in rep and v[0] not in rep:
            rep[f] = v
    if not rep:
        return {}
    copy_area = mesh.face_areas[list(rep.keys())].sum()
    if copy_area < min_share * mesh.face_areas.sum():
        return {}
    return rep

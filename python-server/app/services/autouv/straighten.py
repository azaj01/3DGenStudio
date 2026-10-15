"""Straighten chart borders: re-route each seam run with a cost on turning.

The border min-cut (border.py) decides *where* in a band a border runs -- the
cheapest edges, hidden and sharp. It has no notion of a turn, so on a triangle
mesh the cut it finds can still step from edge to edge in a staircase. Artists
cut along edge loops: long runs that go straight across the surface.

Here every seam *run* (a stretch of chart border between two junctions) is
re-traced by a shortest-path search whose state is the directed edge just
walked, so a step can be charged for how sharply it turns:

    cost = length x (1 - hide_weight x hidden) x (1 + bend x turn / pi)

The search stays in a corridor of ``corridor`` vertex rings around the old run,
never touches another border, and keeps the run's two ends. A new route is
used only if it is cheaper than the old one under the same cost and at least
``min_length`` x as long (shorter means it slipped round the far side of a
feature). The faces between the old and new route then change chart, and the
caller's ``validate`` -- the same re-flattening check the min-cut uses -- has
the final word. Closed rings with no junction are left alone.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _vertex_labels(faces, labels, n_verts):
    """For each vertex, the sorted set of chart labels around it (as a list)."""
    out = [set() for _ in range(n_verts)]
    for f, lab in zip(faces, labels):
        for v in f:
            out[v].add(int(lab))
    return out


def _runs(border_edges, junction):
    """Split border edges into runs between junction vertices. Returns vertex lists."""
    nbr = {}
    for a, b in border_edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen = set()
    runs = []
    for s in nbr:
        if not junction[s]:
            continue
        for nx in nbr[s]:
            e = (min(s, nx), max(s, nx))
            if e in seen:
                continue
            path = [s, nx]
            seen.add(e)
            cur, prev = nx, s
            while not junction[cur]:
                nxt = [w for w in nbr[cur] if w != prev]
                if len(nxt) != 1:
                    break
                w = nxt[0]
                e = (min(cur, w), max(cur, w))
                if e in seen:
                    break
                seen.add(e)
                path.append(w)
                prev, cur = cur, w
            runs.append(path)
    return runs


def straighten_borders(mesh, labels, hidden=None, validate=None, corridor=3, bend=6.0,
                       hide_weight=0.45, min_length=0.7, progress=None):
    """Re-route chart borders along straighter, cheaper paths. Returns ``(labels, stats)``."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    V = mesh.vertices
    faces = mesh.faces
    nV = len(V)
    adj = mesh.adjacency
    aev = mesh.adjacency_edges
    stats = {"runs": 0, "rerouted": 0, "rejected": 0, "unchanged": 0}
    h = np.zeros(nV) if hidden is None else np.asarray(hidden, dtype=np.float64)

    # mesh edges -> vertex adjacency and the faces on each edge
    e_all = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    f_all = np.tile(np.arange(len(faces)), 3)
    ue, inv, cnt = np.unique(e_all, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    open_v = np.zeros(nV, dtype=bool)
    open_v[ue[cnt == 1].ravel()] = True
    G = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(nV, nV)).tocsr()
    G = (G + G.T).tocsr()

    def step_cost(p, u, w):
        d = V[w] - V[u]
        ln = float(np.linalg.norm(d))
        c = ln * (1.0 - hide_weight * 0.5 * (h[u] + h[w]))
        if p is not None:
            d0 = V[u] - V[p]
            n0 = np.linalg.norm(d0)
            if n0 > 0 and ln > 0:
                cosang = float(np.clip(np.dot(d0, d) / (n0 * ln), -1.0, 1.0))
                c *= 1.0 + bend * np.arccos(cosang) / np.pi
        return c, ln

    def path_cost(path):
        tot = ln_tot = 0.0
        for i in range(1, len(path)):
            c, ln = step_cost(path[i - 2] if i >= 2 else None, path[i - 1], path[i])
            tot += c
            ln_tot += ln
        return tot, ln_tot

    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    diff = la != lb
    border = [tuple(e) for e in np.sort(aev[diff], axis=1)]
    vlabs = _vertex_labels(faces, labels, nV)
    bdeg = np.zeros(nV, dtype=np.int64)
    for a, b in border:
        bdeg[a] += 1
        bdeg[b] += 1
    junction = (bdeg != 2) | open_v | np.array([len(s) >= 3 for s in vlabs])
    junction &= bdeg > 0
    runs = _runs(border, junction)
    runs.sort(key=len, reverse=True)
    stats["runs"] = len(runs)
    on_border = bdeg > 0

    for k, run in enumerate(runs):
        if progress is not None:
            progress((k + 1) / max(1, len(runs)))
        if len(run) < 3:
            continue
        s, t = run[0], run[-1]
        # the two charts this run separates (labels may have moved since)
        pair = vlabs[run[len(run) // 2]]
        if len(pair) != 2:
            continue
        A, B = sorted(pair)
        # corridor: vertices within `corridor` rings, only between A and B
        allowed = np.zeros(nV, dtype=bool)
        allowed[run] = True
        for _ in range(corridor):
            allowed |= (G @ allowed.astype(np.float64)) > 0
        cand = np.nonzero(allowed)[0]
        for v in cand:
            if not vlabs[v] <= {A, B}:
                allowed[v] = False
            elif (on_border[v] or open_v[v]) and v not in (s, t) and v not in run:
                allowed[v] = False
        for v in run:
            # other borders that merely touch this run's interior vertices
            if v not in (s, t) and (open_v[v] or len(vlabs[v]) > 2):
                allowed[v] = False
        allowed[s] = allowed[t] = True

        # directed-edge Dijkstra from s to t
        best = {}
        heap = []
        for w in G.indices[G.indptr[s]:G.indptr[s + 1]]:
            if allowed[w]:
                c, _ = step_cost(None, s, w)
                heapq.heappush(heap, (c, s, int(w), None))
        prev_of = {}
        found = None
        pops = 0
        while heap and pops < 200_000:
            c, u, w, back = heapq.heappop(heap)
            pops += 1
            if (u, w) in best:
                continue
            best[(u, w)] = c
            prev_of[(u, w)] = back
            if w == t:
                found = (u, w)
                break
            for x in G.indices[G.indptr[w]:G.indptr[w + 1]]:
                x = int(x)
                if x == u or not allowed[x] or (w, x) in best:
                    continue
                if x == s:
                    continue
                cc, _ = step_cost(u, w, x)
                heapq.heappush(heap, (c + cc, w, x, (u, w)))
        if found is None:
            continue
        new = [found[1]]
        st = found
        while st is not None:
            new.append(st[0])
            st = prev_of[st]
        new.reverse()
        if new == run:
            stats["unchanged"] += 1
            continue
        old_c, old_len = path_cost(run)
        new_c, new_len = path_cost(new)
        if new_c >= old_c - 1e-12 or new_len < min_length * old_len:
            stats["unchanged"] += 1
            continue

        moved = _relabel(mesh, labels, A, B, run, new, allowed)
        if moved is None:
            stats["rejected"] += 1
            continue
        new_labels, changed = moved
        if not changed.any():
            stats["unchanged"] += 1
            continue
        old_a = np.nonzero(labels == A)[0]
        old_b = np.nonzero(labels == B)[0]
        new_a = np.nonzero(new_labels == A)[0]
        new_b = np.nonzero(new_labels == B)[0]
        if len(new_a) == 0 or len(new_b) == 0:
            stats["rejected"] += 1
            continue
        if validate is not None and not validate(old_a, old_b, new_a, new_b):
            stats["rejected"] += 1
            continue
        labels = new_labels
        stats["rerouted"] += 1
        # keep the bookkeeping current for the runs still to come
        for f in np.nonzero(changed)[0]:
            for v in faces[f]:
                vlabs[v] = {int(labels[g]) for g in _faces_of(mesh, v)}
        for a, b in zip(run[:-1], run[1:]):
            bdeg[a] -= 1
            bdeg[b] -= 1
        for a, b in zip(new[:-1], new[1:]):
            bdeg[a] += 1
            bdeg[b] += 1
        on_border = bdeg > 0
    return labels, stats


_VF_CACHE = {}


def _faces_of(mesh, v):
    key = id(mesh)
    vf = _VF_CACHE.get(key)
    if vf is None or vf[0] is not mesh:
        order = np.argsort(mesh.faces.reshape(-1), kind="stable")
        verts_sorted = mesh.faces.reshape(-1)[order]
        starts = np.searchsorted(verts_sorted, np.arange(len(mesh.vertices) + 1))
        vf = (mesh, order // 3, starts)
        _VF_CACHE.clear()
        _VF_CACHE[key] = vf
    _, fo, st = vf
    return fo[st[v]:st[v + 1]]


def _relabel(mesh, labels, A, B, old, new, allowed):
    """Faces of the corridor take the side of the new route they fall on.

    Returns ``(new_labels, changed_mask)`` or None when the new route does not
    separate A from B cleanly (some corridor piece touches both sides).
    """
    faces = mesh.faces
    adj = mesh.adjacency
    aev = np.sort(mesh.adjacency_edges, axis=1)
    inAB = (labels == A) | (labels == B)
    in_corr = inAB & allowed[faces].any(axis=1)
    cut = set(map(tuple, np.sort(np.column_stack([new[:-1], new[1:]]), axis=1).tolist()))
    is_cut = np.array([(int(a), int(b)) in cut for a, b in aev])
    fa, fb = adj[:, 0], adj[:, 1]
    both = in_corr[fa] & in_corr[fb] & ~is_cut
    F = len(faces)
    g = coo_matrix((np.ones(int(both.sum())), (fa[both], fb[both])), shape=(F, F))
    _, comp = connected_components(g, directed=False)
    # which side each corridor piece is glued to (across non-cut edges)
    touch_a = np.zeros(F, dtype=bool)
    touch_b = np.zeros(F, dtype=bool)
    for x, y in ((fa, fb), (fb, fa)):
        edge = in_corr[x] & ~in_corr[y] & inAB[y] & ~is_cut
        ca = comp[x[edge & (labels[y] == A)]]
        cb = comp[x[edge & (labels[y] == B)]]
        touch_a[ca] = True
        touch_b[cb] = True
    new_labels = labels.copy()
    pieces = np.unique(comp[in_corr])
    for c in pieces:
        if touch_a[c] and touch_b[c]:
            return None
        if touch_a[c]:
            new_labels[(comp == c) & in_corr] = A
        elif touch_b[c]:
            new_labels[(comp == c) & in_corr] = B
    return new_labels, new_labels != labels

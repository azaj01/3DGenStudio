"""End-to-end unwrap orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh
from . import segment as _seg
from . import param as _param
from . import postprocess as _post
from . import pack as _pack
from . import weld as _weld
from . import hidden as _hidden
from . import metrics as _metrics
from . import topology as _topology
from . import border as _border
from . import rejoin as _rejoin
from . import ribbons as _ribbons
from . import straighten as _straighten
from . import creases as _creases
from . import planner as _planner
from . import repeats as _repeats
from . import spin as _spin
from . import symmetry as _symmetry


def _split_disconnected(mesh, labels):
    """Give every edge-connected, manifold piece of a chart its own label.

    Segmentation joins faces across any shared edge, including edges shared by
    three or more faces and vertices where two fans merely touch. Once those
    pinches are taken apart (:func:`topology.split_corners`) a chart can fall
    into pieces, and a flattener given several pieces at once has no way to
    place them relative to each other.

    Two degenerate shapes that no cut can open are also given one island per
    face: the faces left over on an edge shared by three or more faces (a fin,
    one layer of a doubled sheet), and tiny closed charts -- typically two
    copies of one triangle with opposite winding, a closed "pillow" whose only
    possible slit is a single edge, which opens nothing.
    """
    out = labels.copy()
    nxt = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(labels[order]))[0] + 1, len(labels)]
    for i in range(len(bounds) - 1):
        fids = order[bounds[i]:bounds[i + 1]]
        f = mesh.faces[fids]
        uniq, inv = np.unique(f.reshape(-1), return_inverse=True)
        lf = inv.reshape(-1, 3)
        lv = mesh.vertices[uniq]
        fn = _topology.face_normals(lv, lf)
        new_lf, orig, comp, n_comp, loose = _topology.split_corners(
            lf, len(lv), fn, return_unpaired=True)
        comp = comp.copy()
        single = np.zeros(len(fids), dtype=bool)
        if len(loose) and (_topology.topology(new_lf, len(orig)).edge_count > 2).any():
            single[loose] = True
            sub = np.nonzero(~single)[0]
            if len(sub):
                # re-derive the pieces of what is left once the loose faces go
                _, _, c2, _ = _topology.split_corners(lf[sub], len(lv), fn[sub])
                comp[sub] = c2
        # tiny closed pieces: no cut opens them, so one island per face
        for k in np.unique(comp[~single]):
            piece = np.nonzero((comp == k) & ~single)[0]
            if len(piece) > 8:
                continue
            p_lf, p_orig, _, _ = _topology.split_corners(lf[piece], len(lv), fn[piece])
            if _topology.topology(p_lf, len(p_orig)).n_loops == 0:
                single[piece] = True
        if single.any():
            comp[single] = int(comp.max()) + 1 + np.arange(int(single.sum()))
        _, comp = np.unique(comp, return_inverse=True)
        comp = np.asarray(comp).ravel()
        for k in range(1, int(comp.max()) + 1):
            out[fids[comp == k]] = nxt
            nxt += 1
    _, out = np.unique(out, return_inverse=True)
    return np.asarray(out).ravel().astype(np.int64)


def _fold_halves(mesh, fids, uv=None, lf=None, seeds="overlap"):
    """Split a folding chart in two along its sharpest crease.

    A chart that folds over itself in every flattening almost always spans a
    ridge or a pocket whose two sides face different ways. Two regions are
    grown over the chart's face graph from its two most opposed faces, and an
    edge costs more to grow across the sharper it is, so the regions meet on a
    crease. (Clustering the face normals instead shatters a noisy AI chart into
    a dozen speckled fragments.)

    With the chart's layout (``uv`` over local faces ``lf``, row-aligned with
    ``fids``) the seeds are the two faces that cover a common texel and lie
    farthest apart on the model -- one on each of the layers that overlap.
    With ``seeds="normals"`` (or no layout) the two most opposed faces are
    used instead; neither choice wins everywhere. Returns a list of face-id
    arrays, one per edge-connected piece, or ``[]`` when the split is
    degenerate.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = mesh.face_normals[fids]
    a = mesh.face_areas[fids]
    local = np.full(mesh.n_faces, -1, dtype=np.int64)
    local[fids] = np.arange(len(fids))
    adj = mesh.adjacency
    inside = (local[adj[:, 0]] >= 0) & (local[adj[:, 1]] >= 0)
    fa, fb = local[adj[inside, 0]], local[adj[inside, 1]]
    if len(fa) == 0:
        return []
    cen = mesh.vertices[mesh.faces[fids]].mean(axis=1)
    crease = np.clip(mesh.adjacency_angle[inside] / (np.pi / 2), 0.0, 1.0)
    w = np.linalg.norm(cen[fa] - cen[fb], axis=1) * (1.0 + 25.0 * crease ** 2) + 1e-12
    G = coo_matrix((w, (fa, fb)), shape=(len(fids), len(fids))).tocsr()
    s0 = s1 = -1
    if uv is not None and lf is not None and seeds == "overlap":
        pairs = _metrics.overlap_pairs(uv, lf)
        if len(pairs):
            d = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
            s0, s1 = (int(x) for x in pairs[int(np.argmax(d))])
    if s0 < 0:
        mean = (n * a[:, None]).sum(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-12)
        s0 = int(np.argmin(n @ mean))
        s1 = int(np.argmin(n @ n[s0]))
    if s0 == s1:
        return []
    _, _, src = dijkstra(G, directed=False, indices=[s0, s1], min_only=True,
                         return_predecessors=True)
    lab = (src == s1).astype(np.int64)
    lab[src < 0] = 0
    if lab.min() == lab.max():
        return []
    pieces = []
    for k in (0, 1):
        sub = fids[lab == k]
        f = mesh.faces[sub]
        uniq, inv = np.unique(f.reshape(-1), return_inverse=True)
        lf = inv.reshape(-1, 3)
        lv = mesh.vertices[uniq]
        _, _, comp, n_comp = _topology.split_corners(lf, len(lv), _topology.face_normals(lv, lf))
        pieces.extend(sub[comp == c] for c in range(n_comp))
    return pieces


def _work_quality(mesh, work):
    """Summary of a finished layout, for choosing between two segmentations.

    ``(islands, shear_share, density_off, overlap_area_share)``: islands is the
    count, shear_share the area sheared past 1.5x, density_off the area whose
    texel density is >25% off once every island is normalised (as packing
    will), and overlap the self-overlapping area.
    """
    verts = mesh.vertices
    tot = float(mesh.face_areas.sum()) or 1.0
    bad = drift = over = 0.0
    for fids, (uv, uniq, lf, info) in work:
        lv = verts[uniq]
        r, a3 = _metrics.face_anisotropy(lv, lf, uv)
        a2 = np.abs(_metrics.signed_uv_area(uv, lf))
        bad += float(a3[r > 1.5].sum())
        s3, s2 = a3.sum(), a2.sum()
        if s3 > 0 and s2 > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.sqrt(a2 / np.where(a3 > 0, a3, np.inf)) * np.sqrt(s3 / s2)
            drift += float(a3[(a3 > 0) & ((d < 0.8) | (d > 1.25))].sum())
        over += float(info["overlap"]) * float(s3)
    return len(work), bad / tot, drift / tot, over / tot


def _looks_organic(mesh, tol_deg=5.0, max_aligned=0.15):
    """A body, not a machine. Gates the character re-plan.

    Counting sharp edges does not work: decimation leaves big dihedral angles
    all over a genuinely smooth surface (43% of an AI character's edge length
    is over 30 degrees). What does separate them is how much surface faces
    exactly along an axis -- walls, floors, panels: 30-60% on buildings and
    props, 2-3% on characters and heads. The model's own principal axes are
    tried too, so a rotated prop still reads as a prop.
    """
    a = mesh.face_areas
    n = mesh.face_normals
    tot = a.sum()
    if tot <= 0:
        return False
    v = mesh.vertices - mesh.vertices.mean(axis=0)
    _, vecs = np.linalg.eigh(v.T @ v)
    best = 0.0
    for frame in (np.eye(3), vecs):
        c = np.abs(n @ frame).max(axis=1)
        best = max(best, float(a[c > np.cos(np.radians(tol_deg))].sum() / tot))
    return best <= max_aligned


def _duplicate_faces(faces):
    """Split faces into keepers and exact duplicates (same three vertices).

    Returns ``(keep_ids, dup_ids, dup_keeper)``: the face ids to unwrap, the
    duplicate face ids, and for each duplicate the id of the face it copies.
    Duplicates are common in AI meshes (a triangle doubled with the opposite
    winding) and poison every chart they land in: they overlap by construction
    and glue edges into three- and four-face fans. They are left out of the
    unwrap and stacked back onto their twin's UVs at the end -- they occupy the
    same place in 3D, so sharing the twin's texels is exactly right.
    """
    key = np.sort(faces, axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    keeper = first[np.asarray(inv).ravel()]
    ids = np.arange(len(faces))
    dup = ids != keeper
    return ids[~dup], ids[dup], keeper[dup]


def _corner_normals(mesh, pre_weld_faces, weld_info, source_normals,
                    preserve_normals, normal_smooth_deg):
    """Resolve one normal per face corner, plus a label saying where it came from.

    Returns ``(corner_group, group_normal, source)``: ``corner_group`` is (F, 3)
    group ids per face corner and ``group_normal`` is (G, 3) unit normals, so a
    corner's normal is ``group_normal[corner_group[f, k]]``. Corners sharing a
    group id are interchangeable, which is what the assembly step keys on.

    **Preferring the input's own normals is the whole point.** Auto UV changes the
    UV layout, not the shape, so it must not change shading either -- and no angle
    heuristic can reconstruct what an artist (or a decimator) authored. Decimated
    organic meshes are the case that proves it: a simplified head measures a
    90th-percentile dihedral around 55 degrees over a surface that is genuinely
    smooth, so any fixed smoothing angle in the usual range shatters it into
    facets while the untouched original renders smooth.

    Carrying them through is exact rather than approximate: the weld maps say
    which input vertex sits behind each welded face corner, so a corner keeps its
    input normal unchanged. Corners are then grouped by normal *value*, which is a
    property of the input geometry alone -- charts never enter into it, so the
    result is automatically continuous across a UV seam, while an authored hard
    edge stays hard because its two sides carry different values.

    Falls back to `Mesh.corner_groups` (dihedral-angle smoothing groups) when the
    input carried no normals, or when the caller asks for a recompute.
    """
    if preserve_normals and source_normals is not None:
        cn = _input_corner_normals(mesh, pre_weld_faces, weld_info, source_normals)
        if cn is not None:
            # Group by normal value. Quantising to a fine grid folds together
            # copies that are equal bar float noise; a pair that straddles a
            # bucket edge merely costs one extra output vertex carrying its own
            # near-identical normal, so the failure mode is harmless.
            flat = cn.reshape(-1, 3)
            q = np.round(flat * 10000.0).astype(np.int64)
            _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
            inv = np.asarray(inv).ravel()
            return inv.reshape(-1, 3), flat[first], "input"

    return (*mesh.corner_groups(normal_smooth_deg), "recomputed")


def _input_corner_normals(mesh, pre_weld_faces, weld_info, source_normals):
    """(F, 3, 3) unit normals per welded face corner, taken from the input mesh.

    None only when the array cannot be lined up with the geometry at all (wrong
    shape, or an index past the end), or when so much of it is degenerate that it
    is not worth trusting -- the caller then falls back to recomputing.

    A handful of zero-length input normals is normal and NOT a reason to discard
    the rest: any vertex whose incident face normals happen to cancel comes out
    zero, and real meshes have a few. Those corners alone fall back to their own
    face normal. (Bailing globally on the first bad corner is a trap worth
    naming: on the reported head mesh exactly 4 corners out of 16836 were
    degenerate, which was enough to throw away every good normal in the file.)
    """
    src = np.asarray(source_normals, dtype=np.float64)
    faces = np.asarray(pre_weld_faces)
    if src.ndim != 2 or src.shape[1] != 3:
        return None

    # Which input vertex is behind each welded face corner. Welding relabels and
    # drops faces but never reorders their corners, so face_index is enough.
    corner_vert = faces[weld_info["face_index"]] if weld_info is not None else faces
    if corner_vert.shape != mesh.faces.shape:
        return None
    if corner_vert.size == 0 or int(corner_vert.max()) >= len(src):
        return None

    cn = src[corner_vert]
    ln = np.linalg.norm(cn, axis=2, keepdims=True)
    bad = ln[..., 0] <= 1e-12
    if bad.mean() > 0.5:  # the channel is junk, not merely imperfect
        return None
    if bad.any():
        # Substitute the corner's own face normal, the best local estimate.
        fn = mesh.face_normals[np.nonzero(bad)[0]]
        cn = cn.copy()
        cn[bad] = fn
        ln = np.linalg.norm(cn, axis=2, keepdims=True)
        ln[ln <= 1e-12] = 1.0
    return cn / ln


@dataclass
class UnwrapResult:
    vertices: np.ndarray          # (Vn,3) output positions (seam-duplicated)
    faces: np.ndarray             # (F,3) indices into vertices
    uv: np.ndarray                # (Vn,2) in [0,1]
    face_chart: np.ndarray        # (F,) chart id per face
    normals: np.ndarray = None    # (Vn,3) normals sampled before the seam split
    stats: dict = field(default_factory=dict)


def unwrap(
    mesh: Mesh,
    max_cone_deg: float = 50.0,
    sharp_weight: float = 0.35,
    min_faces: int = 20,
    min_area_frac: float = 0.004,
    fold_cap_deg: float = 88.0,
    refine: bool = True,
    refine_target_faces: int = 80,
    refine_ad_thresh: float = 1.32,
    resolution: int = 1024,
    padding_texels: int = 4,
    method: str = "auto",
    arap_iters: int = 4,
    weld: bool = True,
    weld_tol_frac: float = 0.1,
    normal_smooth_deg: float = 180.0,
    source_normals=None,
    preserve_normals: bool = True,
    hide_seams: bool = True,
    hide_strength: float = 1.0,
    refine_borders: bool = True,
    border_rings: int = 4,
    straighten_seams: bool = False,
    relax_creases: bool = True,
    character_rescue: bool = True,
    stack_repeats: bool = False,
    orient_up: str = "off",
    unroll_bands: bool = True,
    symmetry: bool = False,
    rejoin: bool = True,
    rejoin_limit: float = 1.45,
    absorb_ribbons: bool = True,
    ensure_disks: bool = True,
    progress=None,
    verbose: bool = True,
) -> UnwrapResult:
    t0 = time.time()

    def report(stage, frac):
        if progress is not None:
            progress(stage, float(frac))

    # ---- topology repair: weld coincident-but-unshared vertices -------------
    # The hard floor on chart count is the number of connected components, so a
    # mesh shattered into hundreds of shells (AI/scan output) is forced to
    # hundreds of charts regardless of how good the parameterisation is. Welding
    # by true distance stitches those shells back into a few components; on
    # already-clean meshes it is a no-op. See autouv.weld.
    comps_before = int(len(np.unique(mesh.components)))
    weld_info = None
    pre_weld_faces = mesh.faces          # needed to trace corners back to input verts
    welded = False
    report("weld", 0.0)
    if weld:
        nv, nf, weld_info = _weld.proximity_weld(
            mesh.vertices, mesh.faces, tol_frac=weld_tol_frac
        )
        if weld_info["verts_after"] != weld_info["verts_before"]:
            mesh = Mesh(nv, nf)
            welded = True
    comps_after = int(len(np.unique(mesh.components)))
    if verbose and weld:
        print(f"[weld] {comps_before} -> {comps_after} components "
              f"({weld_info['verts_before']}->{weld_info['verts_after']} verts, "
              f"tol={weld_info['tol']:.5f})")
    report("weld", 1.0)

    full_mesh = mesh
    keep_ids, dup_ids, dup_keeper = _duplicate_faces(mesh.faces)
    # Every stacked face gets its master's UVs at the end: stack_keeper is the
    # master (full-mesh face id) and stack_perm[j] the master corner that
    # copy corner j lies on. Duplicates match by vertex id.
    fullF = full_mesh.faces
    dup_perm = np.argmax(fullF[dup_ids][:, :, None] == fullF[dup_keeper][:, None, :], axis=2)
    stack = {int(d): (int(k), p) for d, k, p in zip(dup_ids, dup_keeper, dup_perm)}
    n_repeat_faces = 0
    if stack_repeats:
        # Rigid copies and rotational repeats share their master's islands.
        sub = Mesh(full_mesh.vertices, fullF[keep_ids])
        found = _repeats.find_repeats(sub)
        for cf, (mf, p) in found.items():
            stack[int(keep_ids[cf])] = (int(keep_ids[mf]), np.asarray(p))
        n_repeat_faces = len(found)
        # a duplicate of a face that is itself now a copy follows it to the master
        for d in [int(x) for x in dup_ids]:
            k, p = stack[d]
            if k in stack and k not in dup_ids:
                m, pk = stack[k]
                stack[d] = (m, np.asarray(pk)[p])
        if verbose and found:
            print(f"[repeats] {len(found)} faces stacked on their masters")
    if stack:
        stacked = np.zeros(full_mesh.n_faces, dtype=bool)
        stacked[list(stack)] = True
        keep_ids = np.nonzero(~stacked)[0]
        mesh = Mesh(full_mesh.vertices, fullF[keep_ids])
    stack_ids = np.array(sorted(stack), dtype=np.int64)
    stack_keeper = np.array([stack[i][0] for i in stack_ids], dtype=np.int64)
    stack_perm = np.array([stack[i][1] for i in stack_ids], dtype=np.int64).reshape(-1, 3)

    # ---- where seams are least visible -------------------------------------
    # One score per vertex (see autouv.hidden). It prices every seam: the chart
    # borders (border.refine_borders) and any cut made inside a chart
    # (topology.make_disk) both route toward hidden places.
    t_h0 = time.time()
    hidden = None
    vertex_weight = None
    if hide_seams and hide_strength > 0:
        report("hidden", 0.0)
        hidden = _hidden.hidden_field(mesh.vertices, mesh.faces,
                                      progress=lambda p: report("hidden", p))
        vertex_weight = _hidden.vertex_weight(hidden, hide_strength)
        report("hidden", 1.0)
    t_hidden = time.time() - t_h0
    if verbose and hidden is not None:
        print(f"[hidden] {t_hidden:.2f}s (mean {hidden.mean():.2f})")

    labels = _seg.segment(
        mesh,
        max_cone_deg=max_cone_deg,
        sharp_weight=sharp_weight,
        min_faces=min_faces,
        min_area_frac=min_area_frac,
        fold_cap_deg=fold_cap_deg,
    )
    if verbose:
        print(f"[segment] {int(labels.max()) + 1} charts in "
              f"{time.time() - t0:.2f}s")
    report("segment", 1.0)
    if refine:
        labels = _seg.refine_merge(
            mesh, labels,
            target_faces=refine_target_faces,
            ad_thresh=refine_ad_thresh,
            progress=lambda p: report("refine", p),
            method=method,
            arap_iters=arap_iters,
            vertex_weight=vertex_weight,
            make_disk=ensure_disks,
        )

    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.face_areas
    score_cache: dict = {}
    up_axis = {"x": 0, "y": 1, "z": 2}.get(str(orient_up).lower())

    # ---- a protected seam down the mirror plane ------------------------------
    plane = None
    keep_rows = None
    keep_edges = None
    if symmetry:
        plane = _symmetry.find_plane(verts)
        if plane is not None:
            keep_rows = _symmetry.mid_line(mesh, plane)
            if keep_rows.sum() < 8:          # plane does not run along mesh edges
                keep_rows = None
            else:
                ke = np.sort(mesh.adjacency_edges[keep_rows], axis=1)
                keep_edges = set(map(tuple, ke.tolist()))
        if verbose:
            print(f"[symmetry] plane={'none' if plane is None else round(plane[2], 3)} "
                  f"mid-line edges={0 if keep_rows is None else int(keep_rows.sum())}")

    # ---- the flattening helpers every later stage shares ---------------------
    def chart_score(fids):
        key = fids.tobytes()
        hit = score_cache.get(key)
        if hit is None:
            _, _, _, info = _param.parameterize_chart(
                verts, faces, fids, method=method, arap_iters=arap_iters,
                vertex_weight=vertex_weight, make_disk=ensure_disks, fast=True)
            hit = (info["flips"], info["overlap"],
                   info["angle_d"] + 2.0 * info["area_d"], float(areas[fids].sum()))
            score_cache[key] = hit
        return hit

    def keep_move(old_a, old_b, new_a, new_b):
        """Border moves (min-cut, straightening) are kept only when the two
        charts involved flatten at least as well as before."""
        old = [chart_score(old_a), chart_score(old_b)]
        new = [chart_score(new_a), chart_score(new_b)]
        if sum(n[0] for n in new) > sum(o[0] for o in old):
            return False
        worst_old = max(o[1] for o in old)
        if any(n[1] > max(_param.OVERLAP_TOL, worst_old) for n in new):
            return False
        d_old = sum(o[2] * o[3] for o in old) / max(sum(o[3] for o in old), 1e-12)
        d_new = sum(n[2] * n[3] for n in new) / max(sum(n[3] for n in new), 1e-12)
        return d_new <= d_old * 1.01 + 1e-3

    def flatten(fids, cut_pairs=None):
        res = _param.parameterize_chart(
            verts, faces, fids, method=method, arap_iters=arap_iters,
            vertex_weight=vertex_weight, make_disk=ensure_disks, cut_pairs=cut_pairs,
        )
        # Candidates are ranked on a coarse 64-texel raster, which misses thin
        # folded slivers; measure the chosen layout properly before deciding
        # whether it needs splitting.
        uv, _, lf, info = res
        if info["overlap"] > 0 or info["flips"] > 0 or _metrics.boundary_crosses(uv, lf):
            info["overlap"] = _metrics.island_overlap(uv, lf, resolution=256)
        return res

    def flatten_fast(fids):
        # A screen, not a layout: LSCM alone, no fallback search, coarse
        # overlap. rejoin trusts its verdict only on shear, tearing and box
        # growth -- the cases no other flattening could rescue.
        return _param.parameterize_chart(
            verts, faces, fids, method="lscm", arap_iters=0,
            vertex_weight=vertex_weight, make_disk=ensure_disks)

    def flatten_unfolded(fids, depth=0):
        """Flatten a chart; split it by normals while it folds over itself.

        A split is kept only when it lowers the folded area -- a chart whose
        halves fold just as badly stays whole, since more seams would buy
        nothing.
        """
        res = flatten(fids)
        info = res[3]
        if info["overlap"] <= _param.OVERLAP_TOL or depth >= 3 or len(fids) < 16:
            return [(fids, res)]
        before = info["overlap"] * float(areas[fids].sum())
        best = None
        for seeds in ("overlap", "normals"):
            pieces = _fold_halves(mesh, fids, res[0], res[2], seeds=seeds)
            if len(pieces) < 2 or len(pieces) > 6:
                continue
            parts = [(p, flatten(p)) for p in pieces]
            if sum(r[3]["flips"] for _, r in parts) > info["flips"]:
                continue
            after = sum(r[3]["overlap"] * float(areas[p].sum()) for p, r in parts)
            if best is None or after < best[0]:
                best = (after, parts)
        if best is None or best[0] >= 0.8 * before:
            return [(fids, res)]
        out = []
        for p, r in best[1]:
            out.extend(flatten_unfolded(p, depth + 1) if r[3]["overlap"] > _param.OVERLAP_TOL
                       else [(p, r)])
        return out

    def layout(labels, tag="", report=report):
        """Chart labels -> finished islands: border passes, flattening and
        every repair after it. Returns ``(work, stats)`` where ``work`` holds
        ``(fids, (uv, uniq, lf, info))`` per island. ``report`` receives the
        progress of each stage (the re-plan routes it all into one stage)."""
        st = {}
        t_a = time.time()
        # ---- move each chart border onto the cheapest seam near it ----------
        if refine_borders and border_rings > 0 and int(labels.max()) > 0:
            labels, st["borders"] = _border.refine_borders(
                mesh, labels, _border.seam_costs(mesh, vertex_weight),
                rings=int(border_rings), validate=keep_move,
                progress=lambda p: report("borders", p))
        if straighten_seams and int(labels.max()) > 0:
            labels, st["straighten"] = _straighten.straighten_borders(
                mesh, labels, hidden=hidden, validate=keep_move,
                progress=lambda p: report("straighten", p))
        if keep_rows is not None:
            labels = _symmetry.split_by_plane(mesh, labels, plane)
        labels = _split_disconnected(mesh, labels)
        n0 = int(labels.max()) + 1
        t_b = time.time()

        work = []
        for c in range(n0):
            work.extend(flatten_unfolded(np.nonzero(labels == c)[0]))
            report("parameterize", (c + 1) / n0)
        st["fold_splits"] = len(work) - n0

        # ---- slit or split islands that still flatten badly sheared ---------
        if relax_creases:
            work, st["creases"] = _creases.relax_stretched(
                mesh, work, flatten,
                split=lambda fids, res: _fold_halves(mesh, fids, seeds="normals"))
        t_c = time.time()

        # ---- give seams back: re-join neighbours that flatten as well together
        if rejoin and len(work) > 1:
            work, st["rejoin"] = _rejoin.rejoin_islands(
                mesh, work, flatten, flatten_fast=flatten_fast, up_axis=up_axis,
                keep_rows=keep_rows,
                limit=rejoin_limit, floor=rejoin_limit,
                progress=lambda p: report("rejoin", p))
        t_d = time.time()

        # ---- lay thin strip islands back onto their neighbour ----------------
        if absorb_ribbons and len(work) > 1:
            work, st["ribbons"] = _ribbons.absorb_ribbons(mesh, work, keep_edges=keep_edges)
        # ---- treads, pipes, cans: exact rectangles instead of bananas -------
        if unroll_bands:
            work, st["bands_unrolled"] = _spin.unroll_bands(mesh, work)
        st["times"] = {"borders": t_b - t_a, "parameterize": t_c - t_b,
                       "rejoin": t_d - t_c}
        if verbose:
            print(f"[layout{tag}] {len(work)} islands  " + "  ".join(
                f"{k}={v}" for k, v in st.items() if k != "times"))
        return work, st

    t_b0 = time.time()
    work, lay = layout(labels)

    # ---- character re-plan: limbs at the joints, front from back -----------
    # Normal-cone charts on a body leave borders wherever the curvature budget
    # ran out and a long tail of pieces. When the layout came out sheared or
    # fragmented and the model looks organic, cut it the way an artist would
    # (autouv.planner), lay that out with the same passes, keep the better.
    rescue = "off"
    if character_rescue and mesh.n_faces >= 200:
        q0 = _work_quality(mesh, work)
        if (q0[1] > 0.02 or q0[0] >= 60) and _looks_organic(mesh):
            report("rescue", 0.0)
            order = ["borders", "straighten", "parameterize", "rejoin"]

            def rescue_report(stage, frac):
                k = order.index(stage) if stage in order else 0
                report("rescue", (k + float(frac)) / len(order))

            work2, lay2 = layout(_planner.character_labels(mesh), tag="/character",
                                 report=rescue_report)
            q2 = _work_quality(mesh, work2)
            flatter = q2[1] < q0[1]
            simpler = q2[0] <= 0.75 * q0[0] and q2[1] <= max(q0[1], 0.09)
            no_worse = q2[2] <= q0[2] + 0.02 and q2[3] <= q0[3] + 0.002
            rescue = "kept" if (flatter or simpler) and no_worse else "refused"
            if verbose:
                print(f"[rescue] {rescue}: islands {q0[0]}->{q2[0]}  shear {q0[1]:.4f}->{q2[1]:.4f}"
                      f"  drift {q0[2]:.4f}->{q2[2]:.4f}  overlap {q0[3]:.4f}->{q2[3]:.4f}")
            if rescue == "kept":
                work, lay = work2, lay2
            report("rescue", 1.0)
        else:
            rescue = "not_needed"

    border_stats = lay.get("borders")
    straighten_stats = lay.get("straighten")
    n_folds_split = lay["fold_splits"]
    crease_stats = lay.get("creases")
    rejoin_stats = lay.get("rejoin")
    ribbon_stats = lay.get("ribbons")
    t_border = t_b0 + lay["times"]["borders"]
    t_seg = t_border
    t_rejoin = lay["times"]["rejoin"]
    n_charts = len(work)

    # Per-face-corner normals, resolved on the welded topology -- i.e. *before* the
    # seam split below -- so both copies of a seam vertex get the same value and
    # the two sides of every chart boundary shade identically.
    # Resolved on the full welded mesh (duplicates included) so the weld maps
    # still line up; charts index it through keep_ids.
    corner_group, group_normal, normal_source = _corner_normals(
        full_mesh, pre_weld_faces, weld_info if welded else None,
        source_normals, preserve_normals, normal_smooth_deg,
    )

    islands = []          # per-chart uv (local)
    island_uniq = []      # per-chart global vertex ids
    island_faces = []     # per-chart local faces
    island_fids = []      # per-chart global face ids (row-aligned to island_faces)
    island_area3d = []    # per-chart surface area
    angle_ds, area_ds, flips, methods = [], [], [], []
    cut_kinds, cut_edges, not_disk = [], 0, 0

    for c, (fids, (uv, uniq, lf, info)) in enumerate(work):
        if info["cut"] in ("failed", "raw"):
            not_disk += 1
        elif info["cut"] not in ("disk", "none"):
            cut_kinds.append(info["cut"])
            cut_edges += info["cut_edges"]
        if up_axis is None:
            uv = _post.align_island(uv)
        else:
            uv, _ = _post.orient_world_up(uv, verts[uniq], lf, axis=up_axis)
        islands.append(uv)
        island_uniq.append(uniq)
        island_faces.append(lf)
        island_fids.append(fids)
        island_area3d.append(float(mesh.face_areas[fids].sum()))
        angle_ds.append(info["angle_d"])
        area_ds.append(info["area_d"])
        flips.append(info["flips"])
        methods.append(info["method"])

    t_param = time.time()
    if verbose:
        print(f"[param] {n_charts} charts in {t_param - t_seg:.2f}s")

    islands = _post.normalize_texel_density(islands, island_faces, island_area3d)
    packed, fill = _pack.pack(
        islands, resolution=resolution, padding_texels=padding_texels,
        rotate=up_axis is None,
    )
    t_pack = time.time()
    if verbose:
        print(f"[pack] fill={fill:.2%} in {t_pack - t_param:.2f}s")

    # ---- assemble output mesh (duplicate vertices per chart for seam UVs) ----
    out_v = []
    out_src = []          # welded vertex each output vertex was copied from
    out_n = []
    out_uv = []
    out_f = []
    out_fc = []
    row_of_face = np.full(mesh.n_faces, -1, dtype=np.int64)   # face -> out_f row
    voff = 0
    foff = 0
    for c in range(n_charts):
        uniq = island_uniq[c]
        lf = island_faces[c]
        uv = packed[c]

        # An output vertex is one (chart vertex, smoothing group) pair, not just
        # one chart vertex. Splitting only per chart would be enough for a smooth
        # mesh, but a sharp edge running *inside* a chart puts corners from
        # several smoothing groups on the same chart vertex, and collapsing those
        # to one normal re-rounds the edge (a 2-chart cube comes out as a blob).
        # The extra copies are co-located and share the chart vertex's UV, so the
        # UV layout, the face count and the packing are all untouched -- only the
        # vertex count grows, and only where the mesh really is creased.
        #
        # _chart_local builds lf row-aligned to fids, so lf[i, k] and
        # corner_group[fids[i], k] describe the same corner.
        cg = corner_group[keep_ids[island_fids[c]]]            # (Fc, 3) group ids
        corner_key = np.stack([lf.reshape(-1), cg.reshape(-1)], axis=1)
        key_uniq, key_inv = np.unique(corner_key, axis=0, return_inverse=True)
        # numpy >= 2.0 shapes return_inverse after the input when axis is given;
        # flatten so the reshape below is version-independent.
        key_inv = np.asarray(key_inv).ravel()
        local_of_key = key_uniq[:, 0]                           # -> chart vertex
        group_of_key = key_uniq[:, 1]                           # -> smoothing group

        out_v.append(verts[uniq[local_of_key]])
        out_src.append(uniq[local_of_key])
        out_n.append(group_normal[group_of_key])
        out_uv.append(uv[local_of_key])
        out_f.append(key_inv.reshape(-1, 3) + voff)
        out_fc.append(np.full(len(lf), c))
        row_of_face[island_fids[c]] = foff + np.arange(len(lf))
        foff += len(lf)
        voff += len(key_uniq)

    out_v = np.concatenate(out_v, axis=0)
    out_src = np.concatenate(out_src, axis=0)
    out_n = np.concatenate(out_n, axis=0)
    out_uv = np.concatenate(out_uv, axis=0)
    out_f = np.concatenate(out_f, axis=0)
    out_fc = np.concatenate(out_fc, axis=0)

    report("metrics", 0.0)
    quality = _metrics.layout_quality(out_v, out_f, out_uv, out_src, verts, hidden=hidden,
                                      resolution=min(int(resolution), 2048))
    t_metrics = time.time()
    if verbose:
        print(f"[quality] {quality}")

    # ---- stack duplicates and repeats onto their master's UVs ----------------
    # After the metrics on purpose: a stacked copy covers the same texels as
    # its master by design and must not read as overlap.
    if len(stack_ids):
        pos = np.full(full_mesh.n_faces, -1, dtype=np.int64)
        pos[keep_ids] = np.arange(len(keep_ids))
        rows = row_of_face[pos[stack_keeper]]
        kf = out_f[rows]                                   # master's output corners
        src_out = np.take_along_axis(kf, stack_perm, axis=1)
        sfaces = fullF[stack_ids]
        n_new = src_out.size
        # A copy keeps its own position and normal (it sits elsewhere, or a
        # back-to-back twin faces the other way) and takes only the UVs.
        out_v = np.concatenate([out_v, full_mesh.vertices[sfaces].reshape(-1, 3)])
        out_uv = np.concatenate([out_uv, out_uv[src_out].reshape(-1, 2)])
        out_n = np.concatenate([out_n, group_normal[corner_group[stack_ids]].reshape(-1, 3)])
        out_src = np.concatenate([out_src, sfaces.reshape(-1)])
        out_f = np.concatenate([out_f, voff + np.arange(n_new).reshape(-1, 3)])
        out_fc = np.concatenate([out_fc, out_fc[rows]])

    total_flips = int(np.sum(flips))
    stats = {
        "n_faces": int(full_mesh.n_faces),
        "duplicate_faces_stacked": int(len(dup_ids)),
        "repeat_faces_stacked": int(n_repeat_faces),
        "n_charts": n_charts,
        "components_before_weld": comps_before,
        "components_after_weld": comps_after,
        "fill_ratio": float(fill),
        "flipped_triangles": total_flips,
        "mean_angle_distortion": float(np.average(
            angle_ds, weights=island_area3d)),
        "mean_area_distortion": float(np.average(
            area_ds, weights=island_area3d)),
        "method_counts": {m: int(methods.count(m)) for m in set(methods)},
        "normal_source": normal_source,
        "borders_moved": border_stats["moved"] if border_stats else 0,
        "seams_straightened": int(straighten_stats["rerouted"]) if straighten_stats else 0,
        # extra charts made by splitting charts that folded over themselves
        "fold_splits": int(n_folds_split),
        "character_rescue": rescue,
        "bands_unrolled": int(lay.get("bands_unrolled", 0)),
        "symmetry_share": round(float(plane[2]), 4) if plane is not None else 0.0,
        "mid_line_edges": int(keep_rows.sum()) if keep_rows is not None else 0,
        "creases_relaxed": int(crease_stats["relaxed"]) if crease_stats else 0,
        "stretch_splits": int(crease_stats["split"]) if crease_stats else 0,
        "islands_rejoined": int(rejoin_stats["joined"]) if rejoin_stats else 0,
        "ribbons_absorbed": int(ribbon_stats["absorbed"]) if ribbon_stats else 0,
        "charts_cut": len(cut_kinds),
        "cut_edges": int(cut_edges),
        "cut_kinds": {k: cut_kinds.count(k) for k in set(cut_kinds)},
        # charts that stayed non-manifold (sheets glued along an edge) and were
        # flattened as they are
        "charts_not_disk": not_disk,
        "hide_seams": hidden is not None,
        **quality,
        "time_seconds": round(time.time() - t0, 3),
        "time_breakdown": {
            "hidden": round(t_hidden, 3),
            "segment": round(t_b0 - t0 - t_hidden, 3),
            "borders": round(t_border - t_b0, 3),
            "parameterize": round(t_param - t_seg, 3),
            "rejoin": round(t_rejoin, 3),
            "pack": round(t_pack - t_param, 3),
            "metrics": round(t_metrics - t_pack, 3),
        },
    }
    return UnwrapResult(out_v, out_f, out_uv, out_fc, out_n, stats)

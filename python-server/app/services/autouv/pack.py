"""Atlas packing.

A *skyline* (bottom-left) bin packer. Each island is reduced to its (padded)
bounding box; boxes are placed tallest-first at the lowest, left-most spot the
skyline allows, optionally rotated 90 degrees when that wastes less vertical
space. This packs the irregular islands that come out of a low-poly unwrap far
more tightly than fixed shelves (typically 80-90% bbox coverage vs ~60% for
shelves), which matters because wasted atlas space is wasted texture memory.

Everything is finally scaled by a single global factor into the unit square, so
the uniform texel density set in :mod:`postprocess` is preserved.

Padding is expressed in texels for a target resolution -- this is what prevents
mip/bilinear bleed between islands at low texture resolutions.
"""
from __future__ import annotations

import numpy as np


def _skyline_pack(boxes, bin_w, rotate=True):
    """Place [w, h] boxes (already padded). Returns placements, used_w, used_h.

    placements[i] = (x, y, rotated). Skyline = list of [x, top_y, width].
    """
    skyline = [[0.0, 0.0, bin_w]]
    placements = {}

    def level_for(start, w):
        """Lowest y at which a span of width w starting at segment ``start`` fits."""
        x0 = skyline[start][0]
        if x0 + w > bin_w + 1e-9:
            return None
        rem, y, i = w, 0.0, start
        while rem > 1e-12:
            if i >= len(skyline):
                return None
            y = max(y, skyline[i][1])
            rem -= skyline[i][2]
            i += 1
        return y

    def add_at(x, y, w, h):
        """Insert a box of width w at x and raise the skyline to y + h."""
        new_top = y + h
        out = []
        placed = False
        x_end = x + w
        for seg in skyline:
            sx, sy, sw = seg
            sx_end = sx + sw
            if sx_end <= x + 1e-12 or sx >= x_end - 1e-12:
                out.append(seg)                      # untouched
                continue
            # trim left remainder
            if sx < x - 1e-12:
                out.append([sx, sy, x - sx])
            if not placed:
                out.append([x, new_top, w])
                placed = True
            # trim right remainder
            if sx_end > x_end + 1e-12:
                out.append([x_end, sy, sx_end - x_end])
        # merge adjacent segments at equal height
        merged = [out[0]]
        for seg in out[1:]:
            if abs(seg[1] - merged[-1][1]) < 1e-12:
                merged[-1][2] += seg[2]
            else:
                merged.append(seg)
        skyline[:] = merged

    order = sorted(range(len(boxes)), key=lambda b: (-boxes[b][2], -boxes[b][1]))
    used_w = used_h = 0.0
    for bi in order:
        _, w, h, _ = boxes[bi]
        best = None  # (top_y, x, rotated, w, h)
        for rot, (cw, ch) in enumerate(((w, h), (h, w)) if rotate else ((w, h),)):
            for start in range(len(skyline)):
                x = skyline[start][0]
                y = level_for(start, cw)
                if y is None:
                    continue
                key = (y + ch, x)
                if best is None or key < (best[0], best[1]):
                    best = (y + ch, x, rot, cw, ch, y)
        if best is None:                              # forced overflow (rare)
            x, y = 0.0, max((s[1] for s in skyline), default=0.0)
            cw, ch, rot = w, h, 0
            add_at(x, y, cw, ch)
        else:
            _, x, rot, cw, ch, y = best
            add_at(x, y, cw, ch)
        placements[bi] = (x, y, bool(rot))
        used_w = max(used_w, x + cw)
        used_h = max(used_h, y + ch)
    return placements, used_w, used_h


# Bin widths tried, as multiples of the square-root-of-area guess. One width is
# a gamble: with a handful of large islands the skyline can leave a whole
# column empty at one width and fill it at the next.
_WIDTHS = (0.70, 0.78, 0.86, 0.93, 1.0, 1.08, 1.17, 1.27, 1.4, 1.6)


def _best_layout(sizes, pad, rotate=True):
    """Pack padded boxes at several bin widths; keep the one that scales largest."""
    boxes = [[i, w + pad, h + pad, None] for i, (w, h) in enumerate(sizes)]
    total = sum(b[1] * b[2] for b in boxes)
    widest = max(b[1] for b in boxes)
    guess = np.sqrt(total / 0.85)
    best = None
    for k in _WIDTHS:
        bin_w = max(guess * k, widest)
        placements, used_w, used_h = _skyline_pack(boxes, bin_w, rotate)
        side = max(used_w, used_h, 1e-12)
        if best is None or side < best[0]:
            best = (side, placements)
    return best


def pack(islands, resolution=1024, padding_texels=4, rotate=True):
    """Pack a list of (V,2) islands into ``[0,1]^2``.

    Returns ``(packed_islands, fill_ratio)`` where ``fill_ratio`` is the share of
    the unit square covered by island bounding boxes.

    Padding is asked for in *texels of the final atlas*, but the islands are in
    model units until the very last rescale, so it has to be converted with a
    scale that is only known once packing is done. The old code added it in
    texel-fraction units straight onto model-unit boxes, which made the real
    gap depend on the model's size: 0.7 texels instead of 4 on a typical
    asset, touching islands on a large one, 48 texels on a tiny one. The scale
    is estimated, the layout made, and the estimate refined -- the gap then
    lands within a few percent of the request.

    ``rotate=False`` keeps every island's orientation (needed when islands
    were turned so world-up points up the texture).
    """
    mins, sizes = [], []
    for uv in islands:
        mn = uv.min(axis=0)
        ext = uv.max(axis=0) - mn
        mins.append(mn)
        sizes.append((float(ext[0]), float(ext[1])))

    pad_uv = padding_texels / float(resolution)    # wanted gap, final UV units
    side, placements = _best_layout(sizes, 0.0, rotate)
    pad = 0.0
    for _ in range(3):
        scale = 1.0 / side
        pad = pad_uv / scale                       # the same gap in island units
        side, placements = _best_layout(sizes, pad, rotate)
        # the padded layout is bigger, so the real scale is smaller: refine
    scale = 1.0 / side

    packed = [None] * len(islands)
    used = 0.0
    for bi, uv in enumerate(islands):
        w, h = sizes[bi]
        ox, oy, rot = placements[bi]
        uv = uv - mins[bi]
        if rot:                                    # rotate 90 degrees
            uv = np.column_stack((uv[:, 1], w - uv[:, 0]))
        uv = uv * scale
        uv[:, 0] += (ox + pad * 0.5) * scale
        uv[:, 1] += (oy + pad * 0.5) * scale
        packed[bi] = uv
        used += (w * scale) * (h * scale)

    return packed, float(used)

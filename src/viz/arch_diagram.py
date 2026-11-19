"""Paper-style 3D architecture diagrams (feature maps as cuboids, kernels drawn on the input face).

Every layer is drawn as the tensor it OUTPUTS:
  * cuboid height / depth  ~ spatial size H x W  (sqrt scale)
  * cuboid width           ~ number of channels C (log scale)
  * the kernel ("mask") of the operation is drawn as a red square on the H x W face of the
    INPUT tensor, with lines to the output tensor (like the AlexNet / PlotNeuralNet figures)
Below each tensor: name, operation, output shape C x H x W and parameter count.
With draw(..., ops_between=True) the operation (and its stride S / padding P) is written between the two
tensors it connects instead, and only name, shape and parameters stay below each tensor.
Skip connections are arcs above the tensors (add = residual, concat = channel concat,
scale = channel-wise multiply as in squeeze-and-excitation).

Usage: build a list of rows, each row = (title, [Node...], [Skip...]); call draw(rows, path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

DPI = 200
FONT_PT = 14                      # body labels
FONT_PX = FONT_PT * DPI / 72      # ~39 px
CHAR_W = 0.58 * FONT_PX           # average character width (DejaVu Sans)
LINE_H = 1.25 * FONT_PX
GAP = 34                          # space between slots
LEVEL_H = 125                     # vertical distance between nested skip arcs (room for a label)
MARGIN = 40
BG = "#F6F4EF"
INK = "#1B2430"
MUTED = "#5A6270"
KERNEL = ("#E4572E", "#8C2F1B")   # fill, edge

# kind -> (face colour, edge colour)
COLORS = {
    "input": ("#E9E6DD", "#8A8578"),
    "conv": ("#8CC5C1", "#0F6E70"),
    "dw": ("#F2B98A", "#B5541A"),
    "pw": ("#BDB2E3", "#5B4B9A"),
    "pool": ("#D3CEC2", "#7A7466"),
    "se": ("#E9AEBB", "#9E3D52"),
    "fc": ("#6B7B90", "#1B2430"),
    "norm": ("#D3CEC2", "#7A7466"),
}


def _shade(hex_color: str, f: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if f >= 1:   # lighten towards white
        r, g, b = (int(c + (255 - c) * (f - 1)) for c in (r, g, b))
    else:        # darken
        r, g, b = (int(c * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class Node:
    C: int
    H: int
    W: int
    name: str = ""                 # small grey line, e.g. "L2.0 conv1"
    op: str = ""                   # bold line, e.g. "3×3 conv /2"
    params: int | None = None      # parameter count of the operation (None = no line)
    kind: str = "conv"             # colour, see COLORS
    k: int | None = None           # kernel size of the op that produced this node (drawn on the previous node)
    kernel_kind: str = "conv"      # "conv" -> red kernel, "pool" -> grey kernel
    repeat: int = 1                # >1: stacked ghosts + "×N"
    extra: str = ""                # optional extra line (e.g. "24→72→40")
    arrow_only: bool = False       # connect with a plain arrow instead of kernel lines
    sp: str = ""                   # stride / padding, e.g. "S=2, P=1" (shown with the op in ops_between mode)
    act: str = ""                  # what follows the op inside the layer, e.g. "BN + ReLU" (ops_between mode)


@dataclass
class Skip:
    src: int
    dst: int
    label: str = ""
    mode: str = "add"              # add | concat | scale
    post: str = ""                 # operation after the merge, written next to the marker (e.g. "ReLU")


@dataclass
class Scale:
    b1: float = 26.0               # spatial size: b = b1 * sqrt(H)
    a0: float = 14.0               # channel width: a = a0 + a1 * log2(C)
    a1: float = 10.0
    obl: float = 0.42              # oblique (W) length relative to height
    angle: float = 40.0


def _dims(n: Node, S: Scale):
    a = S.a0 + S.a1 * math.log2(max(n.C, 1))
    b = S.b1 * math.sqrt(max(n.H, 1))
    c = S.b1 * math.sqrt(max(n.W, 1)) * S.obl * 2
    dx, dy = c * math.cos(math.radians(S.angle)), c * math.sin(math.radians(S.angle))
    return a, b, dx, dy


def _fmt_params(p):
    return "" if p is None else ("no params" if p == 0 else f"{p:,} params")


def _shape(n: Node):
    return f"{n.C}×{n.H}×{n.W}"


def _op_text(n: Node):
    """Operation lines drawn BETWEEN two tensors (ops_between mode): bold op, then stride/padding."""
    op = n.op + (f"  ×{n.repeat}" if n.repeat > 1 else "")
    return [(t, role) for t, role in ((op, "op"), (n.act, "act"), (n.sp, "sp")) if t]


def _lines(n: Node, ops_between: bool = False):
    op = n.op + (f"  ×{n.repeat}" if n.repeat > 1 else "")
    out = [(n.name, "name")] + ([] if ops_between else [(op, "op")]) + [(_shape(n), "shape")]
    if n.extra:
        out.append((n.extra, "extra"))
    if n.params is not None:
        out.append((_fmt_params(n.params), "params"))
    return out


def _row_geometry(nodes, skips, S, ops_between=False):
    geo, x = [], MARGIN
    for i, n in enumerate(nodes):
        a, b, dx, dy = _dims(n, S)
        ghost = 14 * min(n.repeat - 1, 2)
        body_w = a + dx + ghost
        lab_w = max(len(t) for t, _ in _lines(n, ops_between)) * CHAR_W
        slot = max(body_w, lab_w) + 10
        if ops_between and i > 0:
            # the op text sits centred between this tensor and the previous one: make room for it
            op_w = max((len(t) for t, _ in _op_text(n)), default=0) * CHAR_W * 1.2 + 50
            prev_right = geo[-1]["x"] + geo[-1]["a"] + geo[-1]["dx"]
            body_gap = x + (slot - body_w) / 2 - prev_right
            if op_w > body_gap:
                x += op_w - body_gap
        geo.append(dict(x=x + (slot - body_w) / 2, a=a, b=b, dx=dx, dy=dy, ghost=ghost,
                        cx=x + slot / 2, slot=slot))
        x += slot + GAP
    width = x - GAP + MARGIN
    max_b = max(g["b"] for g in geo)
    top_ext = max(g["b"] / 2 + g["dy"] + g["ghost"] for g in geo)
    n_lines = max(len(_lines(n, ops_between)) for n in nodes)
    label_h = n_lines * LINE_H + 30
    levels = _skip_levels(skips)
    arc_h = (120 + LEVEL_H * max(levels.values())) + (60 + LINE_H * max(s.label.count(chr(10)) for s in skips) if any(s.label for s in skips) else 0) if skips else 30
    height = label_h + _lower(max_b, ops_between) + top_ext + arc_h + 20
    return geo, width, height, label_h, max_b, levels


def _skip_levels(skips):
    levels, placed = {}, []
    for i, s in enumerate(skips):
        lo, hi = sorted((s.src, s.dst))
        lvl = 0
        while any(l == lvl and not (hi <= a or lo >= b) for (a, b, l) in placed):
            lvl += 1
        levels[i] = lvl
        placed.append((lo, hi, lvl))
    return levels


def _cuboid(ax, x, yc, a, b, dx, dy, kind, alpha=1.0, z=2):
    face, edge = COLORS[kind]
    yb, yt = yc - b / 2, yc + b / 2
    ax.add_patch(Rectangle((x, yb), a, b, fc=face, ec=edge, lw=1.6, alpha=alpha, zorder=z))
    ax.add_patch(Polygon([(x, yt), (x + a, yt), (x + a + dx, yt + dy), (x + dx, yt + dy)],
                         closed=True, fc=_shade(face, 1.25), ec=edge, lw=1.6, alpha=alpha, zorder=z))
    ax.add_patch(Polygon([(x + a, yb), (x + a + dx, yb + dy), (x + a + dx, yt + dy), (x + a, yt)],
                         closed=True, fc=_shade(face, 0.82), ec=edge, lw=1.6, alpha=alpha, zorder=z))


def _face_point(g, yc, u, v):
    """Point on the right (H x W) face: u along W (oblique, 0..1), v along H (0..1)."""
    xr, yb = g["x"] + g["a"], yc - g["b"] / 2
    return (xr + u * g["dx"], yb + v * g["b"] + u * g["dy"])


def _lower(max_b, ops_between):
    """space between the row's centre line and the labels below (room for the op text in the gaps)"""
    return max(max_b / 2, 3 * LINE_H + 40) if ops_between else max_b / 2


def _draw_row(ax, title, nodes, skips, S, y0, geo, label_h, max_b, levels, ops_between=False):
    yc = y0 + label_h + _lower(max_b, ops_between)
    if title:
        ax.text(MARGIN, y0 - 6, title, fontsize=FONT_PT + 2, fontweight="bold", color=INK, va="top")
    # z-order: cuboid i sits at Z(i); the kernel + lines of node i sit just above cuboid i-1 (on its face)
    # and below cuboid i (so the lines disappear behind the output tensor).
    Z = lambda i: 5 + 0.02 * i
    for i in range(1, len(nodes)):
        n, g, gp, prev = nodes[i], geo[i], geo[i - 1], nodes[i - 1]
        target = (g["x"] + g["a"] * 0.5, yc)
        zk = Z(i - 1) + 0.01
        if n.k and not n.arrow_only:
            fh = max(min(1.0, n.k / prev.H), 14 / gp["b"])
            fw = max(min(1.0, n.k / prev.W), 12 / max(gp["dx"], 1))
            u0, v0 = 0.5 - fw / 2, 0.55 - fh / 2
            corners = [_face_point(gp, yc, u, v) for u, v in
                       ((u0, v0), (u0 + fw, v0), (u0 + fw, v0 + fh), (u0, v0 + fh))]
            kf, ke = KERNEL if n.kernel_kind != "pool" else ("#8A8578", "#3D4756")
            for c in corners:
                ax.plot([c[0], target[0]], [c[1], target[1]], color=ke, lw=1.1, alpha=0.75, zorder=zk)
            ax.add_patch(Polygon(corners, closed=True, fc=kf, ec=ke, lw=1.6, alpha=1.0, zorder=zk + 0.001))
        else:
            xs = gp["x"] + gp["a"] + gp["dx"] * 0.5 + 6
            ax.annotate("", xy=(g["x"] - 4, yc), xytext=(xs, yc),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.0, mutation_scale=22), zorder=1)
    # cuboids (ghosts for repeated blocks behind their main cuboid)
    for i, (n, g) in enumerate(zip(nodes, geo)):
        for r in range(min(n.repeat - 1, 2), 0, -1):
            _cuboid(ax, g["x"] + 14 * r, yc + 14 * r, g["a"], g["b"], g["dx"], g["dy"], n.kind, alpha=0.45,
                    z=Z(i) - 0.005)
        _cuboid(ax, g["x"], yc, g["a"], g["b"], g["dx"], g["dy"], n.kind, z=Z(i))
    # labels below
    y_lab = yc - _lower(max_b, ops_between) - 26
    for n, g in zip(nodes, geo):
        for j, (t, role) in enumerate(_lines(n, ops_between)):
            style = dict(name=dict(color=MUTED, fontsize=FONT_PT - 1),
                         op=dict(color=INK, fontsize=FONT_PT, fontweight="bold"),
                         shape=dict(color=INK, fontsize=FONT_PT),
                         extra=dict(color=MUTED, fontsize=FONT_PT - 1),
                         params=dict(color="#0F6E70", fontsize=FONT_PT - 1))[role]
            ax.text(g["cx"], y_lab - j * LINE_H, t, ha="center", va="top", zorder=6, **style)
    # skip connections above
    tops = [yc + g["b"] / 2 + g["dy"] + g["ghost"] for g in geo]
    if ops_between:
        # operation between tensor i-1 and tensor i, written in the gap under the kernel lines
        for i in range(1, len(nodes)):
            lines = _op_text(nodes[i])
            if not lines:
                continue
            xm = (geo[i - 1]["x"] + geo[i - 1]["a"] + geo[i - 1]["dx"] + geo[i]["x"]) / 2
            for j, (t, role) in enumerate(lines):
                style = dict(op=dict(color=INK, fontsize=FONT_PT, fontweight="bold"),
                             act=dict(color="#5B4B9A", fontsize=FONT_PT - 1),
                             sp=dict(color="#8C2F1B", fontsize=FONT_PT - 1))[role]
                ax.text(xm, yc - 22 - j * LINE_H, t, ha="center", va="top", zorder=9, **style)
    base = max(tops) + 70
    for i, s in enumerate(skips):
        gs, gd = geo[s.src], geo[s.dst]
        xs, xe = gs["x"] + (gs["a"] + gs["dx"]) / 2, gd["x"] + (gd["a"] + gd["dx"]) / 2
        ys, ye_node = tops[s.src] + 6, tops[s.dst] + 6
        marker_y = ye_node + 34
        peak = base + LEVEL_H * levels[i]
        style = "--" if s.mode == "concat" else "-"
        path = Path([(xs, ys), (xs, peak), (xe, peak), (xe, marker_y + 22)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, fc="none", ec=INK, lw=2.4, ls=style, zorder=7))
        sym = {"add": "+", "concat": "‖", "scale": "×"}[s.mode]
        ax.add_patch(Circle((xe, marker_y), 22, fc=BG, ec=INK, lw=2.2, zorder=8))
        ax.text(xe, marker_y - 1, sym, ha="center", va="center", fontsize=FONT_PT + 2, fontweight="bold",
                color=INK, zorder=9)
        ax.plot([xe, xe], [marker_y - 22, ye_node], color=INK, lw=2.2, zorder=7)
        if s.post:
            ax.text(xe + 30, marker_y, s.post, ha="left", va="center", fontsize=FONT_PT - 1,
                    color="#5B4B9A", fontweight="bold", zorder=9, bbox=dict(fc=BG, ec="none", pad=1))
        if s.label:
            ax.text((xs + xe) / 2, peak + 10, s.label, ha="center", va="bottom",
                    fontsize=FONT_PT - 1, color=INK, zorder=9,
                    bbox=dict(fc=BG, ec="none", pad=2))


def draw(rows, path, scale: Scale | None = None, caption: str = "", ops_between: bool = False):
    """rows: list of (title, nodes, skips), first row drawn at the top.
    Writes a PNG at DPI 200 and returns its pixel size (width, height)."""
    S = scale or Scale()
    laid = []
    for title, nodes, skips in rows:
        geo, w, h, label_h, max_b, levels = _row_geometry(nodes, skips, S, ops_between)
        laid.append(dict(title=title, nodes=nodes, skips=skips, geo=geo, w=w, h=h,
                         label_h=label_h, max_b=max_b, levels=levels,
                         title_h=FONT_PX * 1.8 if title else 0))
    width = max(r["w"] for r in laid)
    cap_h = (LINE_H * (caption.count("\n") + 1) + 20) if caption else 0
    total_h = sum(r["h"] + r["title_h"] for r in laid) + cap_h + 2 * MARGIN
    fig = plt.figure(figsize=(width / DPI, total_h / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(0, total_h); ax.axis("off")
    if caption:
        ax.text(MARGIN, MARGIN, caption, fontsize=FONT_PT - 1, color=MUTED, va="bottom")
    y = MARGIN + cap_h                      # y grows upwards: draw the LAST row first, at the bottom
    for r in reversed(laid):
        _draw_row(ax, None, r["nodes"], r["skips"], S, y, r["geo"], r["label_h"], r["max_b"], r["levels"],
                  ops_between)
        if r["title"]:
            ax.text(MARGIN, y + r["h"] + r["title_h"] * 0.35, r["title"], fontsize=FONT_PT + 2,
                    fontweight="bold", color=INK, va="bottom")
        y += r["h"] + r["title_h"]
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return int(width), int(total_h)

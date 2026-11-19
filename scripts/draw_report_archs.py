"""Paper-style 3D diagrams of the trained architectures at the 200 px input used in the report.

Same renderer as scripts/draw_standard_archs.py (src/viz/arch_diagram.py); shapes come from forward
hooks on the real models and parameter counts from the modules, so the figures match the code.

Figures (results/figures/arch_report/):
  own_cnn_final_part1/2   our CNN, 2 basic blocks per stage, every conv
  own_cnn_bottleneck      the ablation block (stage 2, bottleneck) next to the basic block it replaces
  resnet18_part1/2        ResNet18 with the 1-output head, every conv
  mobilenet_v3_large      MobileNetV3-Large with the 1-output head, one cuboid per block

Usage:  python scripts/draw_report_archs.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as M
from torchvision.models import mobilenetv3

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from viz.arch_diagram import Node, Scale, Skip, draw  # noqa: E402
from src.models.own_cnn import OwnCNN  # noqa: E402

OUT = REPO / "results" / "figures" / "arch_report"
OUT.mkdir(parents=True, exist_ok=True)
RES = 200
X = torch.zeros(1, 3, RES, RES)
INPUT = Node(3, RES, RES, name="input", op="RGB image", kind="input")


def trace(model, modules):
    """One forward pass; returns {label: output shape (C, H, W)} for the given modules."""
    shapes, hooks = {}, []
    for label, mod in modules.items():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, label=label: shapes.__setitem__(label, (tuple(o.shape[1:]) + (1, 1))[:3])))
    with torch.inference_mode():
        model(X)
    for h in hooks:
        h.remove()
    return shapes


def P(mod):
    return sum(p.numel() for p in mod.parameters())


def conv_op(conv):
    k, s = conv.kernel_size[0], conv.stride[0]
    dw = conv.groups > 1 and conv.groups == conv.in_channels
    return f"{k}×{k} {'dw ' if dw else ''}conv"


def conv_sp(conv):
    """stride and padding as written between the tensors, e.g. 'S=2, P=1'"""
    return f"S={conv.stride[0]}, P={conv.padding[0]}"


def save(name, rows, caption, scale=None):
    w, h = draw(rows, OUT / f"{name}.png", scale=scale, caption=caption, ops_between=True)
    print(f"{name:24s} {w} x {h} px")


# ------------------------------------------------------------------ own CNN, final (2 basic blocks per stage)
m = OwnCNN(block="basic", depths=(2, 2, 2, 2)).eval()
stem_conv = m.stem[0][0]
mods = {"stem": m.stem[0], "pool": m.stem[1], "gap": m.pool, "fc": m.fc}
for si in range(1, 5):
    for bi, blk in enumerate(getattr(m, f"stage{si}")):
        mods[f"S{si}.{bi}.c1"], mods[f"S{si}.{bi}.c2"] = blk.main[0], blk.main[1]
        if not isinstance(blk.shortcut, nn.Identity):
            mods[f"S{si}.{bi}.sc"] = blk.shortcut
S = trace(m, mods)


def oconv(label):
    conv = mods[label][0]
    act = "BN + ReLU" if label.endswith("c1") else "BN only"
    return Node(*S[label], name=label.replace(".c", " conv"), op=conv_op(conv), sp=conv_sp(conv), act=act,
                params=conv.weight.numel(), kind="conv", k=conv.kernel_size[0])


def proj(label):
    return f"projection shortcut\n1×1 conv + BN, S=2, P=0 · {mods[label][0].weight.numel():,} params"


n1 = [INPUT,
      Node(*S["stem"], name="stem: conv", op="3×3 conv", act="BN + ReLU", sp="S=2, P=1", params=stem_conv.weight.numel(), k=3),
      Node(*S["pool"], name="stem: maxpool", op="3×3 max pool", sp="S=2, P=1", params=0, kind="pool", k=3, kernel_kind="pool")]
for si in (1, 2):
    for bi in (0, 1):
        n1 += [oconv(f"S{si}.{bi}.c1"), oconv(f"S{si}.{bi}.c2")]
k1 = [Skip(2, 4, "identity shortcut", post="ReLU"), Skip(4, 6, "identity shortcut", post="ReLU"), Skip(6, 8, proj("S2.0.sc"), post="ReLU"), Skip(8, 10, "identity shortcut", post="ReLU")]
save("own_cnn_final_part1", [("Our CNN (final) at 200 px, part 1: stem, stage 1, stage 2", n1, k1)],
     "BN = BatchNorm. In each residual block the second conv has no ReLU of its own: the ReLU comes after the + (shown next to it).\n"
     "Between two tensors: the operation that maps one to the next (S = stride, P = padding). Red square = its kernel. Params = conv weights only.")

n2 = [Node(*S["S2.1.c2"], name="from stage 2", op="S2.1 conv2 output", kind="conv")]
for si in (3, 4):
    for bi in (0, 1):
        n2 += [oconv(f"S{si}.{bi}.c1"), oconv(f"S{si}.{bi}.c2")]
n2 += [Node(*S["gap"], name="avgpool", op="global avg pool", params=0, kind="pool", arrow_only=True),
       Node(*S["fc"], name="fc (head)", op="linear 256→1", extra="age = 33.23 + 19.9·out", params=P(m.fc),
            kind="fc", arrow_only=True)]
k2 = [Skip(0, 2, proj("S3.0.sc"), post="ReLU"), Skip(2, 4, "identity shortcut", post="ReLU"), Skip(4, 6, proj("S4.0.sc"), post="ReLU"), Skip(6, 8, "identity shortcut", post="ReLU")]
save("own_cnn_final_part2", [("Our CNN (final) at 200 px, part 2: stage 3, stage 4, head", n2, k2)],
     f"Total: {P(m):,} parameters (conv + BatchNorm + head); 374.1 M MACs per image; receptive field 431 px.")

# ------------------------------------------------------------------ ablation: bottleneck block vs basic block (stage 2)
mb = OwnCNN(block="bottleneck").eval()
bb, bs = mb.stage2, m.stage2[0]
mods_b = {"in": mb.stage1, "c1": bb.main[0], "c2": bb.main[1], "c3": bb.main[2]}
mods_s = {"in": m.stage1, "c1": bs.main[0], "c2": bs.main[1]}
Sb, Ss = trace(mb, mods_b), trace(m, mods_s)
basic = [Node(*Ss["in"], name="stage 1 output", op="block input", kind="conv"),
         Node(*Ss["c1"], name="conv1", op="3×3 conv", act="BN + ReLU", sp="S=2, P=1", params=bs.main[0][0].weight.numel(), kind="conv", k=3),
         Node(*Ss["c2"], name="conv2", op="3×3 conv", act="BN only", sp="S=1, P=1", params=bs.main[1][0].weight.numel(), kind="conv", k=3)]
bott = [Node(*Sb["in"], name="stage 1 output", op="block input", kind="conv"),
        Node(*Sb["c1"], name="conv1", op="1×1 conv", act="BN + ReLU", sp="S=1, P=0", extra="squeeze to C/4", params=bb.main[0][0].weight.numel(),
             kind="pw", k=1),
        Node(*Sb["c2"], name="conv2", op="3×3 conv", act="BN + ReLU", sp="S=2, P=1", params=bb.main[1][0].weight.numel(), kind="conv", k=3),
        Node(*Sb["c3"], name="conv3", op="1×1 conv", act="BN only", sp="S=1, P=0", extra="expand to C", params=bb.main[2][0].weight.numel(),
             kind="pw", k=1)]
save("own_cnn_bottleneck",
     [("Main design: basic block (stage 2, first block)", basic,
       [Skip(0, 2, f"projection shortcut\n1×1 conv + BN, S=2, P=0 · {bs.shortcut[0].weight.numel():,} params", post="ReLU")]),
      ("Ablation: bottleneck block in the same place", bott,
       [Skip(0, 3, f"projection shortcut\n1×1 conv + BN, S=2, P=0 · {bb.shortcut[0].weight.numel():,} params", post="ReLU")])],
     "Params = conv weights only.")

# ------------------------------------------------------------------ ResNet18 (every conv)
m = M.resnet18(); m.fc = nn.Linear(512, 1); m.eval()
mods = {"conv1": m.conv1, "maxpool": m.maxpool, "avgpool": m.avgpool, "fc": m.fc}
for li in range(1, 5):
    for bi, blk in enumerate(getattr(m, f"layer{li}")):
        mods[f"L{li}.{bi}.c1"], mods[f"L{li}.{bi}.c2"] = blk.conv1, blk.conv2
        if blk.downsample is not None:
            mods[f"L{li}.{bi}.down"] = blk.downsample[0]
S = trace(m, mods)


def rconv(label):
    conv = mods[label]
    act = "BN + ReLU" if label.endswith("c1") else "BN only"
    return Node(*S[label], name=label.replace(".c", " conv"), op=conv_op(conv), sp=conv_sp(conv), act=act,
                params=conv.weight.numel(), kind="conv", k=conv.kernel_size[0])


def down(label):
    return f"projection shortcut\n1×1 conv + BN, S=2, P=0 · {mods[label].weight.numel():,} params"


n1 = [INPUT,
      Node(*S["conv1"], name="stem: conv1", op="7×7 conv", act="BN + ReLU", sp="S=2, P=3", params=m.conv1.weight.numel(), k=7),
      Node(*S["maxpool"], name="stem: maxpool", op="3×3 max pool", sp="S=2, P=1", params=0, kind="pool", k=3, kernel_kind="pool")]
for li in (1, 2):
    for bi in (0, 1):
        n1 += [rconv(f"L{li}.{bi}.c1"), rconv(f"L{li}.{bi}.c2")]
k1 = [Skip(2, 4, "identity shortcut", post="ReLU"), Skip(4, 6, "identity shortcut", post="ReLU"), Skip(6, 8, down("L2.0.down"), post="ReLU"), Skip(8, 10, "identity shortcut", post="ReLU")]
save("resnet18_part1", [("ResNet18 at 200 px, part 1: stem, layer1, layer2", n1, k1)],
     "BN = BatchNorm. In each residual block the second conv has no ReLU of its own: the ReLU comes after the + (shown next to it).\n"
     "Between two tensors: the operation that maps one to the next (S = stride, P = padding). Red square = its kernel. Params = conv weights only.")
n2 = [Node(*S["L2.1.c2"], name="from layer2", op="L2.1 conv2 output", kind="conv")]
for li in (3, 4):
    for bi in (0, 1):
        n2 += [rconv(f"L{li}.{bi}.c1"), rconv(f"L{li}.{bi}.c2")]
n2 += [Node(*S["avgpool"], name="avgpool", op="global avg pool", params=0, kind="pool", arrow_only=True),
       Node(*S["fc"], name="fc (head)", op="linear 512→1", params=P(m.fc), kind="fc", arrow_only=True)]
k2 = [Skip(0, 2, down("L3.0.down"), post="ReLU"), Skip(2, 4, "identity shortcut", post="ReLU"), Skip(4, 6, down("L4.0.down"), post="ReLU"), Skip(6, 8, "identity shortcut", post="ReLU")]
save("resnet18_part2", [("ResNet18 at 200 px, part 2: layer3, layer4, head", n2, k2)],
     "Total: 11,177,025 parameters; 1.556 G MACs per image at 200 px. ImageNet weights, new 1-output head.")

# ------------------------------------------------------------------ MobileNetV3-Large
m = M.mobilenet_v3_large(); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1); m.eval()
cfg, _ = mobilenetv3._mobilenet_v3_conf("mobilenet_v3_large")
f = m.features
mods = {f"f{i}": f[i] for i in range(len(f))}
mods["pool"] = m.avgpool
S = trace(m, mods)
nodes = [INPUT, Node(*S["f0"], name="stem: conv", op="3×3 conv", act="BN + h-swish", sp="S=2, P=1", params=P(f[0]), kind="conv", k=3)]
skips = []
for i, c in enumerate(cfg, start=1):
    op = f"block k{c.kernel}" + (" SE" if c.use_se else "")
    nodes.append(Node(*S[f"f{i}"], name=f"B{i} ({'HS' if c.use_hs else 'RE'})", op=op, act="h-swish inside" if c.use_hs else "ReLU inside", sp=f"S={c.stride}",
                      extra=f"{c.input_channels}→{c.expanded_channels}→{c.out_channels}",
                      params=P(f[i]), kind="pw", k=c.kernel))
    if f[i].use_res_connect:
        skips.append(Skip(len(nodes) - 2, len(nodes) - 1))
last, hidden = len(f) - 1, m.classifier[0]
nodes += [Node(*S[f"f{last}"], name="last conv", op="1×1 conv", act="BN + h-swish", sp="S=1, P=0", params=P(f[last]), kind="pw", k=1),
          Node(*S["pool"], name="avgpool", op="global avg pool", params=0, kind="pool", arrow_only=True),
          Node(hidden.out_features, 1, 1, name="fc1", op=f"linear {hidden.in_features}→{hidden.out_features}",
               extra="h-swish, dropout 0.2", params=P(hidden), kind="fc", arrow_only=True),
          Node(1, 1, 1, name="fc2 (head)", op=f"linear {hidden.out_features}→1", params=P(m.classifier[-1]),
               kind="fc", arrow_only=True)]
split = 10
rows = [(f"MobileNetV3-Large at 200 px, part {p}", nodes[lo:hi],
         [Skip(s.src - lo, s.dst - lo) for s in skips if lo <= s.src and s.dst < hi])
        for p, lo, hi in ((1, 0, split), (2, split, len(nodes)))]
save("mobilenet_v3_large", rows,
     "Each block: 1×1 expand → k×k depthwise (stride S) → (SE) → 1×1 project. Red square = the depthwise kernel. "
     "Block params include BatchNorm and SE.\n+ = identity shortcut (no activation after the add). Total 4,203,313 parameters; 187 M MACs at 200 px.",
     scale=Scale(b1=22))
print("done ->", OUT.relative_to(REPO))

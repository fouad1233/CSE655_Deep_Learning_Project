"""Draw paper-style 3D diagrams of the candidate standard models at a 128 px input.

All shapes come from forward hooks on the real torchvision models and all parameter
counts from the modules themselves, so the figures cannot drift from the code.

Usage:  python scripts/draw_standard_archs.py            -> results/figures/arch/*.png
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as M
from torchvision.models import efficientnet, mobilenetv3

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from viz.arch_diagram import Node, Scale, Skip, draw  # noqa: E402

OUT = REPO / "results" / "figures" / "arch"
OUT.mkdir(parents=True, exist_ok=True)
RES = 128
X = torch.zeros(1, 3, RES, RES)


def trace(model, modules):
    """Run one forward pass and return {label: output shape (C, H, W)} for the given modules."""
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


def conv_op(conv, prefix=""):
    k, s = conv.kernel_size[0], conv.stride[0]
    dw = conv.groups > 1 and conv.groups == conv.in_channels
    return f"{prefix}{k}×{k} {'dw ' if dw else ''}conv" + (f" /{s}" if s > 1 else "")


INPUT = Node(3, RES, RES, name="input", op="RGB image", kind="input")
written = {}


def save(name, rows, caption, scale=None):
    w, h = draw(rows, OUT / f"{name}.png", scale=scale, caption=caption)
    written[name] = (w, h)
    print(f"{name:28s} {w} x {h} px")


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
    return Node(*S[label], name=label.replace(".c", " conv"), op=conv_op(conv),
                params=conv.weight.numel(), kind="conv", k=conv.kernel_size[0])


def down_label(label):
    c = mods[label]
    return f"shortcut: 1×1 conv /2 · {c.weight.numel():,} params"


n1 = [INPUT,
      Node(*S["conv1"], name="conv1", op="7×7 conv /2", params=m.conv1.weight.numel(), k=7),
      Node(*S["maxpool"], name="maxpool", op="3×3 max /2", params=0, kind="pool", k=3, kernel_kind="pool")]
for li in (1, 2):
    for bi in (0, 1):
        n1 += [rconv(f"L{li}.{bi}.c1"), rconv(f"L{li}.{bi}.c2")]
k1 = [Skip(2, 4, "identity"), Skip(4, 6, "identity"), Skip(6, 8, down_label("L2.0.down")), Skip(8, 10, "identity")]
save("resnet18_part1", [("ResNet18 at 128 px, part 1: stem, layer1, layer2", n1, k1)],
     "Every conv is followed by BatchNorm + ReLU (the block's last ReLU comes after the +). "
     "Red square = the conv kernel on the tensor it slides over. Params = conv weights only.")

prev = S["L2.1.c2"]
n2 = [Node(*prev, name="from layer2", op="L2.1 conv2 output", kind="conv")]
for li in (3, 4):
    for bi in (0, 1):
        n2 += [rconv(f"L{li}.{bi}.c1"), rconv(f"L{li}.{bi}.c2")]
n2 += [Node(*S["avgpool"], name="avgpool", op="global avg", params=0, kind="pool", arrow_only=True),
       Node(*S["fc"], name="fc (head)", op="linear 512→1", params=P(m.fc), kind="fc", arrow_only=True)]
k2 = [Skip(0, 2, down_label("L3.0.down")), Skip(2, 4, "identity"), Skip(4, 6, down_label("L4.0.down")),
      Skip(6, 8, "identity")]
save("resnet18_part2", [("ResNet18 at 128 px, part 2: layer3, layer4, head", n2, k2)],
     "Total: 11,166,912 conv + 9,600 BatchNorm + 513 head = 11,177,025 parameters; 592.2 M MACs per image.")

# ------------------------------------------------------------------ ResNet50 (stages + one bottleneck)
m = M.resnet50(); m.fc = nn.Linear(2048, 1); m.eval()
b = m.layer2[0]
mods = {"conv1": m.conv1, "maxpool": m.maxpool, "avgpool": m.avgpool, "fc": m.fc,
        **{f"layer{i}": getattr(m, f"layer{i}") for i in range(1, 5)},
        "b.c1": b.conv1, "b.c2": b.conv2, "b.c3": b.conv3, "b.in": m.layer1}
S = trace(m, mods)
macro = [INPUT,
         Node(*S["conv1"], name="conv1", op="7×7 conv /2", params=m.conv1.weight.numel(), k=7),
         Node(*S["maxpool"], name="maxpool", op="3×3 max /2", params=0, kind="pool", k=3, kernel_kind="pool")]
for i, reps in zip(range(1, 5), (3, 4, 6, 3)):
    macro.append(Node(*S[f"layer{i}"], name=f"layer{i}", op="bottleneck", repeat=reps,
                      params=P(getattr(m, f"layer{i}")), kind="conv", k=3))
macro += [Node(*S["avgpool"], name="avgpool", op="global avg", params=0, kind="pool", arrow_only=True),
          Node(*S["fc"], name="fc (head)", op="linear 2048→1", params=P(m.fc), kind="fc", arrow_only=True)]
detail = [Node(*S["b.in"], name="layer1 output", op="block input", kind="conv"),
          Node(*S["b.c1"], name="conv1", op="1×1 conv", params=b.conv1.weight.numel(), kind="pw", k=1),
          Node(*S["b.c2"], name="conv2", op="3×3 conv /2", params=b.conv2.weight.numel(), kind="conv", k=3),
          Node(*S["b.c3"], name="conv3", op="1×1 conv", params=b.conv3.weight.numel(), kind="pw", k=1)]
save("resnet50", [("ResNet50: stem, 4 stages of bottleneck blocks, head", macro, []),
                  ("One bottleneck in detail: layer2, block 0 (downsampling)", detail,
                   [Skip(0, 3, f"shortcut: 1×1 conv /2 · {b.downsample[0].weight.numel():,} params")])],
     "Stage params include BatchNorm. Detail params = conv weights only. Stride sits on the 3×3 conv (ResNet V1.5).",
     scale=Scale(b1=22))

# ------------------------------------------------------------------ ConvNeXt-Tiny
m = M.convnext_tiny(); m.classifier[-1] = nn.Linear(768, 1); m.eval()
f = m.features
mods = {f"f{i}": f[i] for i in range(8)}
mods["pool"] = m.avgpool
S = trace(m, mods)
names = ["stem", "stage1", "down1", "stage2", "down2", "stage3", "down3", "stage4"]
macro = [INPUT]
for i, nm in enumerate(names):
    if nm == "stem":
        macro.append(Node(*S["f0"], name="stem", op="4×4 conv /4", params=P(f[0]), kind="conv", k=4))
    elif nm.startswith("down"):
        macro.append(Node(*S[f"f{i}"], name=nm, op="LN + 2×2 conv /2", params=P(f[i]), kind="conv", k=2))
    else:
        macro.append(Node(*S[f"f{i}"], name=nm, op="CNBlock", repeat=len(f[i]), params=P(f[i]), kind="dw", k=7))
macro += [Node(*S["pool"], name="avgpool + LN", op="global avg", params=P(m.classifier[0]), kind="pool",
               arrow_only=True),
          Node(1, 1, 1, name="fc (head)", op="linear 768→1", params=P(m.classifier[-1]), kind="fc", arrow_only=True)]
blk = f[1][0]
dwc, lin1, lin2 = blk.block[0], blk.block[3], blk.block[5]
C = dwc.in_channels
detail = [Node(C, 32, 32, name="stage1 input", op="block input", kind="dw"),
          Node(C, 32, 32, name="dwconv", op="7×7 dw conv", params=P(dwc), kind="dw", k=7),
          Node(lin1.out_features, 32, 32, name="pwconv1", op=f"linear {C}→{lin1.out_features}",
               extra="LayerNorm before, GELU after", params=P(lin1), kind="pw", k=1),
          Node(C, 32, 32, name="pwconv2", op=f"linear {lin1.out_features}→{C}",
               extra=f"× layer-scale γ ({blk.layer_scale.numel()} params)", params=P(lin2), kind="pw", k=1)]
save("convnext_tiny", [("ConvNeXt-Tiny: patchify stem, 4 stages (3/3/9/3 blocks), head", macro, []),
                       ("One CNBlock in detail (stage1, 96 channels, 32×32)", detail, [Skip(0, 3, "identity")])],
     "Stage params include LayerNorm and layer-scale. The two linear layers act per pixel, i.e. as 1×1 convs "
     "(output shapes shown as C×H×W).", scale=Scale(b1=22))


# ------------------------------------------------------------------ MobileNetV3 (Large and Small)
def mbv3(variant, split):
    ctor = M.mobilenet_v3_large if variant == "large" else M.mobilenet_v3_small
    m = ctor(); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1); m.eval()
    cfg, _ = mobilenetv3._mobilenet_v3_conf(f"mobilenet_v3_{variant}")
    f = m.features
    mods = {f"f{i}": f[i] for i in range(len(f))}
    mods["pool"] = m.avgpool
    S = trace(m, mods)
    nodes = [INPUT, Node(*S["f0"], name="stem", op="3×3 conv /2", params=P(f[0]), kind="conv", k=3)]
    skips = []
    for i, c in enumerate(cfg, start=1):
        op = f"k{c.kernel}" + (" SE" if c.use_se else "") + (" /2" if c.stride == 2 else "")
        nodes.append(Node(*S[f"f{i}"], name=f"B{i} ({'HS' if c.use_hs else 'RE'})", op=op,
                          extra=f"{c.input_channels}→{c.expanded_channels}→{c.out_channels}",
                          params=P(f[i]), kind="pw", k=c.kernel))
        if f[i].use_res_connect:
            skips.append(Skip(len(nodes) - 2, len(nodes) - 1))
    last = len(f) - 1
    hidden = m.classifier[0]
    nodes += [Node(*S[f"f{last}"], name="last conv", op="1×1 conv", params=P(f[last]), kind="pw", k=1),
              Node(*S["pool"], name="avgpool", op="global avg", params=0, kind="pool", arrow_only=True),
              Node(hidden.out_features, 1, 1, name="fc1", op=f"linear {hidden.in_features}→{hidden.out_features}",
                   extra="h-swish, dropout 0.2", params=P(hidden), kind="fc", arrow_only=True),
              Node(1, 1, 1, name="fc2 (head)", op=f"linear {hidden.out_features}→1", params=P(m.classifier[-1]),
                   kind="fc", arrow_only=True)]
    # split into two rows; skips re-indexed per row
    rows = []
    for title, lo, hi in ((f"MobileNetV3-{variant.capitalize()}, part 1", 0, split),
                          (f"MobileNetV3-{variant.capitalize()}, part 2", split, len(nodes))):
        rows.append((title, nodes[lo:hi], [Skip(s.src - lo, s.dst - lo) for s in skips if lo <= s.src and s.dst < hi]))
    return m, rows


m, rows = mbv3("large", 10)
save("mobilenet_v3_large", rows,
     "Each block: 1×1 expand → k×k depthwise → (SE) → 1×1 project. Red square = the block's depthwise kernel. "
     "Block params include BatchNorm and SE. + = residual (stride 1, same channels).", scale=Scale(b1=22))
_, rows = mbv3("small", 8)
save("mobilenet_v3_small", rows,
     "Same block as MobileNetV3-Large, 11 blocks. Block params include BatchNorm and SE.", scale=Scale(b1=22))

# one MobileNetV3-Large block in detail: block 5 (40→120→40, k5, SE, stride 1)
blk = m.features[5]
exp_c, dw_c, se, proj = blk.block[0][0], blk.block[1][0], blk.block[2], blk.block[3][0]
mods = {"in": m.features[4], "exp": blk.block[0], "dw": blk.block[1], "se_pool": se.avgpool,
        "se1": se.fc1, "se2": se.fc2, "se": se, "proj": blk.block[3]}
S = trace(m, mods)
detail = [Node(*S["in"], name="block 5 input", op="from block 4", kind="pw"),
          Node(*S["exp"], name="expand", op="1×1 conv", extra="BN + ReLU", params=exp_c.weight.numel(), kind="pw", k=1),
          Node(*S["dw"], name="depthwise", op="5×5 dw conv", extra="BN + ReLU", params=dw_c.weight.numel(), kind="dw", k=5),
          Node(*S["se_pool"], name="SE squeeze", op="global avg", params=0, kind="pool", arrow_only=True),
          Node(*S["se1"], name="SE fc1", op="1×1 conv", extra="ReLU", params=P(se.fc1), kind="se", arrow_only=True),
          Node(*S["se2"], name="SE fc2", op="1×1 conv", extra="hard-sigmoid", params=P(se.fc2), kind="se", arrow_only=True),
          Node(*S["se"], name="SE excite", op="scale channels", params=0, kind="dw", arrow_only=True),
          Node(*S["proj"], name="project", op="1×1 conv", extra="BN, no activation", params=proj.weight.numel(),
               kind="pw", k=1)]
save("mobilenet_v3_block", [("One MobileNetV3-Large block in detail: block 5 (40→120→40, 5×5, SE, stride 1)", detail,
                             [Skip(2, 6, "multiply by the 120 SE weights", mode="scale"), Skip(0, 7, "identity")])],
     "Params = conv weights (+ bias for the SE layers); BatchNorm adds 2 per channel. "
     "SE squeeze width = make_divisible(120 / 4, 8) = 32.")

# ------------------------------------------------------------------ EfficientNet-B0
m = M.efficientnet_b0(); m.classifier[-1] = nn.Linear(1280, 1); m.eval()
f = m.features
blocks = [(si, b) for si in range(1, 8) for b in f[si]]
mods = {"stem": f[0], "head": f[8], "pool": m.avgpool, **{f"b{j}": b for j, (_, b) in enumerate(blocks, start=1)}}
S = trace(m, mods)
cfg, _ = efficientnet._efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
nodes = [INPUT, Node(*S["stem"], name="stem", op="3×3 conv /2", params=P(f[0]), kind="conv", k=3)]
skips = []
for j, (si, b) in enumerate(blocks, start=1):
    c = cfg[si - 1]
    conv_dw = b.block[1][0] if c.expand_ratio != 1 else b.block[0][0]
    k, s = conv_dw.kernel_size[0], conv_dw.stride[0]
    nodes.append(Node(*S[f"b{j}"], name=f"B{j} (stage {si})", op=f"MBConv{c.expand_ratio} k{k}" + (" /2" if s == 2 else ""),
                      params=P(b), kind="pw", k=k))
    if b.use_res_connect:
        skips.append(Skip(len(nodes) - 2, len(nodes) - 1))
nodes += [Node(*S["head"], name="head conv", op="1×1 conv", params=P(f[8]), kind="pw", k=1),
          Node(*S["pool"], name="avgpool", op="global avg", params=0, kind="pool", arrow_only=True),
          Node(1, 1, 1, name="fc (head)", op="linear 1280→1", extra="dropout 0.2 before", params=P(m.classifier[-1]),
               kind="fc", arrow_only=True)]
split = 10
rows = [("EfficientNet-B0, part 1", nodes[:split], [s for s in skips if s.dst < split]),
        ("EfficientNet-B0, part 2", nodes[split:], [Skip(s.src - split, s.dst - split) for s in skips if s.src >= split])]
save("efficientnet_b0", rows,
     "MBConv = 1×1 expand → k×k depthwise → SE → 1×1 project, SiLU activations. Red square = depthwise kernel. "
     "Block params include BatchNorm and SE.", scale=Scale(b1=22))

# ------------------------------------------------------------------ ShuffleNetV2 1.0x
m = M.shufflenet_v2_x1_0(); m.fc = nn.Linear(1024, 1); m.eval()
u = m.stage3[1]
mods = {"conv1": m.conv1, "maxpool": m.maxpool, "stage2": m.stage2, "stage3": m.stage3, "stage4": m.stage4,
        "conv5": m.conv5, "u.in": m.stage3[0], "u.pw1": u.branch2[2], "u.dw": u.branch2[4], "u.pw2": u.branch2[7],
        "u.out": u}
S = trace(m, mods)
macro = [INPUT,
         Node(*S["conv1"], name="conv1", op="3×3 conv /2", params=P(m.conv1), kind="conv", k=3),
         Node(*S["maxpool"], name="maxpool", op="3×3 max /2", params=0, kind="pool", k=3, kernel_kind="pool")]
for nm, reps in (("stage2", 4), ("stage3", 8), ("stage4", 4)):
    macro.append(Node(*S[nm], name=nm, op="shuffle unit", repeat=reps, params=P(getattr(m, nm)), kind="dw", k=3))
macro += [Node(*S["conv5"], name="conv5", op="1×1 conv", params=P(m.conv5), kind="pw", k=1),
          Node(S["conv5"][0], 1, 1, name="avgpool", op="global avg", params=0, kind="pool", arrow_only=True),
          Node(1, 1, 1, name="fc (head)", op="linear 1024→1", params=P(m.fc), kind="fc", arrow_only=True)]
half = S["u.in"][0] // 2
detail = [Node(*S["u.in"], name="unit input", op=f"split {half} | {half}", kind="dw"),
          Node(*S["u.pw1"], name="right half", op="1×1 conv", extra="BN + ReLU", params=u.branch2[0].weight.numel(),
               kind="pw", k=1),
          Node(*S["u.dw"], name="", op="3×3 dw conv", extra="BN", params=u.branch2[3].weight.numel(), kind="dw", k=3),
          Node(*S["u.pw2"], name="", op="1×1 conv", extra="BN + ReLU", params=u.branch2[5].weight.numel(), kind="pw", k=1),
          Node(*S["u.out"], name="unit output", op="concat + shuffle", params=0, kind="dw", arrow_only=True)]
save("shufflenet_v2", [("ShuffleNetV2 1.0×: stem, 3 stages of shuffle units (4/8/4), head", macro, []),
                       ("One shuffle unit in detail (stage3, stride 1, 232 channels at 8×8)", detail,
                        [Skip(0, 4, f"left half: {half} channels pass through unchanged", mode="concat")])],
     "Stage params include BatchNorm. Detail params = conv weights only. "
     "The first unit of each stage has stride 2 and processes both halves.", scale=Scale(b1=22))

print("done:", len(written), "figures in", OUT.relative_to(REPO))

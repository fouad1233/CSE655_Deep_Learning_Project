"""Measure params, MACs, output shapes and CPU latency of the candidate standard models (torchvision).

Usage:  python scripts/model_facts.py results/model_facts.json
Numbers feed docs/report_notes.md and the "Standard CNN Architectures" slide deck.
"""
import json
import sys
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.flop_counter import FlopCounterMode
from torchvision import models as M

out_path = sys.argv[1]
torch.manual_seed(0)
torch.set_num_threads(4)

SPECS = {
    "resnet18": (M.resnet18, M.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (M.resnet50, M.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (M.convnext_tiny, M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (M.mobilenet_v3_large, M.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (M.mobilenet_v3_small, M.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (M.efficientnet_b0, M.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (M.shufflenet_v2_x1_0, M.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
}


def head_to_regression(name, m):
    """Replace the 1000-class layer by a 1-output linear layer; return (description, in_features)."""
    if name.startswith("resnet") or name.startswith("shufflenet"):
        f = m.fc.in_features; m.fc = nn.Linear(f, 1); return "model.fc", f
    if name.startswith("mobilenet") or name.startswith("efficientnet") or name.startswith("convnext"):
        f = m.classifier[-1].in_features; m.classifier[-1] = nn.Linear(f, 1); return "model.classifier[-1]", f
    raise ValueError(name)


def macs(model, res):
    x = torch.zeros(1, 3, res, res)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2          # FlopCounterMode counts 2 FLOPs per multiply-accumulate


def latency_ms(model, res, reps=20):
    x = torch.randn(1, 3, res, res)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
    return 1000 * (time.perf_counter() - t) / reps


facts = {"torch": torch.__version__, "torchvision": torchvision.__version__, "models": {}}
for name, (ctor, w) in SPECS.items():
    meta = w.meta
    m = ctor(weights=None).eval()
    orig_head = repr(m.fc if hasattr(m, "fc") else m.classifier)
    n_params_1000 = sum(p.numel() for p in m.parameters())
    macs224_1000 = macs(m, 224)
    where, feat = head_to_regression(name, m)
    m.eval()
    n_params_reg = sum(p.numel() for p in m.parameters())
    # stage output shapes at 128 px (top-level feature blocks)
    shapes = []
    x = torch.zeros(1, 3, 128, 128)
    blocks = (list(m.features) if hasattr(m, "features") else
              [m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4] if name.startswith("resnet") else
              [m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5])
    with torch.inference_mode():
        for b in blocks:
            x = b(x)
            shapes.append([type(b).__name__, list(x.shape[1:])])
    facts["models"][name] = {
        "weights": str(w), "acc1": meta["_metrics"]["ImageNet-1K"]["acc@1"], "acc5": meta["_metrics"]["ImageNet-1K"]["acc@5"],
        "num_params_meta": meta["num_params"], "gflops_meta": meta.get("_ops"), "file_size_mb_meta": meta.get("_file_size"),
        "recipe": meta.get("recipe"), "min_size": meta.get("min_size"),
        "params_1000cls": n_params_1000, "params_regression": n_params_reg,
        "gmacs_224_measured": round(macs224_1000 / 1e9, 4),
        "gmacs_128_regression": round(macs(m, 128) / 1e9, 4),
        "gmacs_64_regression": round(macs(m, 64) / 1e9, 4),
        "latency_ms_cpu_128_bs1": round(latency_ms(m, 128), 1),
        "head_replaced": where, "head_in_features": feat, "original_head": orig_head,
        "stage_shapes_128": shapes,
    }
    print(name, "done", flush=True)

# architecture configuration tables, read from torchvision's own config helpers
from torchvision.models import mobilenetv3, efficientnet, convnext
cfg = {}
for v in ["mobilenet_v3_large", "mobilenet_v3_small"]:
    setting, last = mobilenetv3._mobilenet_v3_conf(v)
    cfg[v] = {"blocks": [dict(in_c=c.input_channels, k=c.kernel, exp=c.expanded_channels, out=c.out_channels,
                              se=bool(c.use_se), act="HS" if c.use_hs else "RE", s=c.stride) for c in setting],
              "last_channel": last}
setting, last = efficientnet._efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
cfg["efficientnet_b0"] = {"stages": [dict(expand=c.expand_ratio, k=c.kernel, s=c.stride, in_c=c.input_channels,
                                          out=c.out_channels, layers=c.num_layers) for c in setting], "last_channel": last}
m = M.convnext_tiny()
cfg["convnext_tiny"] = {"stages": [[int(st[0].block[0].in_channels), len(st)] for st in m.features[1::2]]}
m = M.shufflenet_v2_x1_0()
cfg["shufflenet_v2_x1_0"] = {"stage_out_channels": m._stage_out_channels,
                             "repeats": [len(m.stage2), len(m.stage3), len(m.stage4)]}
facts["configs"] = cfg

# ResNet18 full per-conv table at 128 px (regression head)
m = M.resnet18(); m.fc = nn.Linear(512, 1); m.eval()
rows = []
def hook(mod, inp, out, name=None):
    k = mod.kernel_size[0]; cin = mod.in_channels; cout = mod.out_channels
    h, w = out.shape[2:]
    rows.append(dict(name=name, k=k, s=mod.stride[0], p=mod.padding[0], cin=cin, cout=cout,
                     out=[cout, h, w], params=sum(p.numel() for p in mod.parameters()),
                     macs=k * k * cin * cout * h * w))
for n, mod in m.named_modules():
    if isinstance(mod, nn.Conv2d):
        mod.register_forward_hook(lambda mod, i, o, n=n: hook(mod, i, o, n))
with torch.inference_mode():
    m(torch.zeros(1, 3, 128, 128))
facts["resnet18_convs_128"] = rows
facts["resnet18_bn_params"] = sum(p.numel() for mod in m.modules() if isinstance(mod, nn.BatchNorm2d) for p in mod.parameters())

json.dump(facts, open(out_path, "w"), indent=1)
print("saved", out_path)

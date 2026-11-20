"""Our own CNN for age regression, trained from scratch (input 3 x 200 x 200, output 1 number = age).

Design. The table below is the FIRST version (v1, depths = (1, 1, 1, 1), chosen 2026-11-18; hand calculations in
notebook 02 section 15). The FINAL model uses depths = (2, 2, 2, 2), see the note after the table.

    stem     conv3x3  3->32, stride 2   -> 32 x 100 x 100     RF   3   jump 2
             maxpool 3x3, stride 2      -> 32 x  50 x  50     RF   7   jump 4
    stage1   block 32->32,  stride 1    -> 32 x  50 x  50     RF  23   jump 4
    stage2   block 32->64,  stride 2    -> 64 x  25 x  25     RF  47   jump 8
    stage3   block 64->128, stride 2    -> 128 x 13 x 13      RF  95   jump 16
    stage4   block 128->256, stride 2   -> 256 x  7 x  7      RF 191   jump 32
    head     global average pool -> 256 -> Linear(256, 1) -> age = mean + std * output (fixed scaling)

    FINAL: depths = (2, 2, 2, 2) (variant "own_cnn_d2", 2026-11-19): every stage gets a second block with stride 1 and an
    identity shortcut (same layout as ResNet-18 at half its widths, so ~1/4 of its params). RF after each stage: 39, 95, 207, 431.

    block = "basic"      : conv3x3 -> BN -> ReLU -> conv3x3 -> BN, + shortcut, ReLU      (main design)
    block = "bottleneck" : conv1x1 (C/4) -> conv3x3 (C/4) -> conv1x1 (C), + shortcut   (ablation)
    shortcut = identity if the shape stays the same, else conv1x1 (stride 2) + BN ("projection")

Formulas used everywhere below:
    output size  = floor((H + 2*padding - kernel) / stride) + 1
    conv params  = C_in * C_out * k * k            (bias=False: the BatchNorm after it has its own shift)
    conv MACs    = H_out * W_out * C_out * C_in * k * k
    BN params    = 2 * C                            (gamma = scale, beta = shift)
    receptive field: RF_new = RF_old + (k - 1) * jump,  jump_new = jump * stride
"""
import torch
import torch.nn as nn


def conv_bn(c_in: int, c_out: int, k: int, stride: int, relu: bool = True) -> nn.Sequential:
    """Conv (k x k, 'same' padding k//2, no bias) -> BatchNorm -> optional ReLU."""
    layers = [nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=k // 2, bias=False),
              nn.BatchNorm2d(c_out)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """output = ReLU( F(x) + S(x) )   F = main path, S = shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int, kind: str = "basic"):
        super().__init__()
        if kind == "basic":
            # two 3x3 convs; the first one does the stride (halves H and W when stride = 2)
            self.main = nn.Sequential(conv_bn(c_in, c_out, 3, stride),
                                      conv_bn(c_out, c_out, 3, 1, relu=False))   # ReLU comes after the addition
        elif kind == "bottleneck":
            mid = c_out // 4                         # 1x1 squeezes to a quarter of the channels
            self.main = nn.Sequential(conv_bn(c_in, mid, 1, 1),            # 1x1: mix channels, C_in -> C/4
                                      conv_bn(mid, mid, 3, stride),        # 3x3 on the thin tensor (the only RF growth)
                                      conv_bn(mid, c_out, 1, 1, relu=False))  # 1x1: back up to C_out
        else:
            raise ValueError(kind)
        # the two paths must have the same shape to be added:
        # same shape -> identity (0 params); otherwise 1x1 conv with the same stride + BN (projection)
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = conv_bn(c_in, c_out, 1, stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class OwnCNN(nn.Module):
    def __init__(self, block: str = "basic", widths=(32, 64, 128, 256), depths=(1, 1, 1, 1)):
        super().__init__()
        w1, w2, w3, w4 = widths

        def stage(c_in, c_out, stride, n):
            """n residual blocks: the first changes the shape (stride / channels), the others keep it (identity shortcut).
            n = 1 -> the block itself (keeps the weight names of the first trained models: stage1.main...)."""
            blocks = [ResidualBlock(c_in, c_out, stride, block)] + [ResidualBlock(c_out, c_out, 1, block) for _ in range(n - 1)]
            return blocks[0] if n == 1 else nn.Sequential(*blocks)

        # stem: shrink 200 -> 100 (strided conv) -> 50 (max pool) early, where the tensors are biggest
        self.stem = nn.Sequential(conv_bn(3, w1, 3, 2),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        d1, d2, d3, d4 = depths
        self.stage1 = stage(w1, w1, 1, d1)    # 50 x 50, identity shortcut
        self.stage2 = stage(w1, w2, 2, d2)    # 25 x 25
        self.stage3 = stage(w2, w3, 2, d3)    # 13 x 13
        self.stage4 = stage(w3, w4, 2, d4)    # 7 x 7
        self.pool = nn.AdaptiveAvgPool2d(1)              # 256 x 7 x 7 -> 256 x 1 x 1 (mean of the 49 positions)
        self.fc = nn.Linear(w4, 1)                       # the regression head: 256 weights + 1 bias
        # output scaling (fixed, NOT learned; buffers are saved in the checkpoint but get no gradient):
        #   age = out_mean + out_std * fc(features)
        # fc predicts the age in standard deviations ("z-score"). train_model sets out_mean / out_std to the
        # training-set mean (33.23) and std (~19.9). Without it the head weights, which start at 0 and move
        # by ~lr per Adam step, needed hundreds of steps to spread predictions over +-20 years
        # (diagnostic in notebook 02 section 15.3.1: prediction std 5-7 vs true 19.9 after 310 steps).
        self.register_buffer("out_mean", torch.tensor(0.0))
        self.register_buffer("out_std", torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming (He) init: keeps the activation variance constant through ReLU layers
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        # last BN of every main path starts at gamma = 0 -> F(x) = 0 -> each block starts as
        # ReLU(shortcut(x)): the untrained network behaves like a shallow one, which makes the
        # first epochs from scratch stable (same idea as torchvision's zero_init_residual=True)
        for blk in self.modules():
            if isinstance(blk, ResidualBlock):
                nn.init.zeros_(blk.main[-1][1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        x = torch.flatten(self.pool(x), 1)               # (B, 256, 1, 1) -> (B, 256)
        return self.out_mean + self.out_std * self.fc(x)  # (B, 1), in years

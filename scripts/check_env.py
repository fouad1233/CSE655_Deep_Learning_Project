"""Print the environment so every machine can be compared (and logged in the report).

Usage:  python scripts/check_env.py
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["torch", "torchvision", "timm", "codecarbon", "numpy", "pandas",
            "matplotlib", "scikit-learn", "pillow", "torchinfo", "pyyaml", "tqdm"]


def main() -> int:
    print(f"python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform : {platform.platform()} | {platform.machine()} | {platform.processor()}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"venv     : {'yes' if in_venv else 'NO - activate .venv first!'}")

    for pkg in PACKAGES:
        try:
            print(f"  {pkg:<13} {version(pkg)}")
        except PackageNotFoundError:
            print(f"  {pkg:<13} MISSING")

    import torch
    print(f"cuda     : {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" | build {torch.version.cuda} | archs {torch.cuda.get_arch_list()}")
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            print(f"  gpu{i}: {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")
            if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
                print("  WARNING: this torch build has no kernels for this GPU - reinstall with a newer CUDA index")
    else:
        print()
    mps = getattr(torch.backends, "mps", None)
    print(f"mps      : {bool(mps and mps.is_available())}")

    # tiny smoke test on the best available device
    dev = "cuda" if torch.cuda.is_available() else "mps" if (mps and mps.is_available()) else "cpu"
    x = torch.randn(8, 3, 64, 64, device=dev)
    y = torch.nn.Conv2d(3, 4, 3, padding=1).to(dev)(x)
    assert y.shape == (8, 4, 64, 64)
    print(f"smoke    : conv2d OK on {dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

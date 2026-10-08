"""Phase 0 environment check.

Run from the repository root with the venv active:
    python check_env.py

It prints the version of every library the project depends on and confirms
that PyTorch can see the NVIDIA GPU. If CUDA is not available, training
would silently fall back to the CPU and take days instead of hours.
"""

import sys
from importlib.metadata import version, PackageNotFoundError

PACKAGES = [
    "torch", "torchvision", "ultralytics", "fastapi", "uvicorn",
    "streamlit", "opencv-python", "numpy", "pandas", "matplotlib",
]


def main() -> int:
    print(f"{'Python':<15}{sys.version.split()[0]}  ({sys.executable})")
    missing = []
    for name in PACKAGES:
        try:
            print(f"{name:<15}{version(name)}")
        except PackageNotFoundError:
            print(f"{name:<15}NOT INSTALLED")
            missing.append(name)

    import torch

    print()
    print(f"CUDA build of PyTorch : {torch.version.cuda}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: PyTorch cannot see the GPU. Is the CPU-only build installed?")
        return 1

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU                   : {gpu.name}, {gpu.total_memory / 1024**3:.1f} GB")

    # A tiny matrix multiply on the GPU proves CUDA actually runs,
    # not just that the driver was detected.
    x = torch.rand(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    print(f"GPU test computation  : OK ({y:.1f})")

    if missing:
        print(f"\nFAIL: missing packages: {', '.join(missing)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

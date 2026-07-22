from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Compiled CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA. Check the NVIDIA driver and confirm "
            "that the installed PyTorch build includes '+cu126'."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"Total VRAM: {properties.total_memory / 2**30:.2f} GiB")

    x = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()

    print(f"Matrix output shape: {tuple(y.shape)}")
    print(f"Matrix output dtype: {y.dtype}")
    print("CUDA test passed.")


if __name__ == "__main__":
    main()

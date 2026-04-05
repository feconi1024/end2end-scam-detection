from __future__ import annotations

from typing import Optional

import torch


def resolve_runtime_device(requested_device: str, verbose: bool = True) -> str:
    """
    Resolve the runtime device without relying on torch.cuda.is_available().

    On some cluster setups, calling `torch.cuda.is_available()` can trigger a
    fragile cudaGetDeviceCount path that emits warnings and incorrectly causes
    a CPU fallback. Instead, we try a tiny allocation on the requested device.
    """

    device_str = str(requested_device).lower()
    if not device_str.startswith("cuda"):
        return device_str

    if not torch.backends.cuda.is_built():
        if verbose:
            print("PyTorch was built without CUDA support; using device='cpu'.")
        return "cpu"

    try:
        probe = torch.empty(1, device=torch.device(device_str))
        del probe
        return device_str
    except Exception as exc:
        if verbose:
            print(
                f"CUDA initialization failed for device='{device_str}'; "
                f"using device='cpu'. Error: {exc}"
            )
        return "cpu"


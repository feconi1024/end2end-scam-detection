from __future__ import annotations

import torch


def resolve_runtime_device(requested_device: str, verbose: bool = True) -> str:
    """
    Normalize the requested runtime device without eagerly touching CUDA.

    We intentionally avoid `torch.cuda.is_available()` and any test allocation
    here. Both can trigger fragile early CUDA initialization on some clusters
    and lead to false CPU fallbacks. If the user requests CUDA, we keep that
    request and let the real `model.to(device)` path either succeed or raise a
    clear error instead of silently training on CPU for hours.
    """

    device_str = str(requested_device).lower()
    if device_str.startswith("cuda"):
        if verbose:
            if torch.backends.cuda.is_built():
                print(f"Requested runtime device: '{device_str}'.")
            else:
                print(
                    "Requested runtime device starts with 'cuda', but this "
                    "PyTorch build reports no CUDA support. Initialization "
                    "will fail unless you switch training.device to 'cpu'."
                )
        return device_str

    if verbose:
        print(f"Requested runtime device: '{device_str}'.")
    return device_str


import torch

def _check_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return
    raise ValueError(
        f"expected torch.Tensor, got {type(obj)}"
    )
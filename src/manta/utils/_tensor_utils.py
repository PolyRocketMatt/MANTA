import torch

def _check_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return
    raise ValueError(
        f"unexpected type: {type(obj)}, expected torch.Tensor"
    )
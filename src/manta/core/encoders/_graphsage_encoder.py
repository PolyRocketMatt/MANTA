import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass 
class _GraphRepresentation:
    x: torch.Tensor     # [N, d] - feature matrix
    P: torch.Tensor     # [N, N] - transition matrix


@dataclass
class _GraphBatch:
    x: torch.Tensor     # [B, d] - feature matrix
    P: torch.Tensor     # [B, B] transition matrix
    idx: torch.Tensor   # [B] - original graph indices for each node



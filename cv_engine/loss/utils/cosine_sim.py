import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustCosineSim(nn.Module):
    def forward(
            self,
            feat1: torch.Tensor,
            feat2: torch.Tensor,
            dim: int = -1,
            eps: float = 1.0e-8,
            clamp: float = 1.0e2
    ):
        feat1 = feat1.clamp(-clamp, clamp)
        feat2 = feat2.clamp(-clamp, clamp)
        sim = F.cosine_similarity(feat1, feat2, dim=dim, eps=eps)
        return sim


class PairwiseCosineSim(nn.Module):
    def forward(
            self,
            feat1: torch.Tensor,
            feat2: torch.Tensor,
            eps: float = 1.0e-8,
            clamp: float = 1.0e2
    ) -> torch.Tensor:
        feat1 = feat1.clamp(-clamp, clamp)
        feat2 = feat2.clamp(-clamp, clamp)
        inner = torch.inner(feat1, feat2)
        norm1 = torch.norm(feat1, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(feat2, p=2, dim=1, keepdim=True)
        sim = inner / (norm1 * norm2.T + eps)
        return sim

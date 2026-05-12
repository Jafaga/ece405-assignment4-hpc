from __future__ import annotations

import math
import torch


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            q_idx = torch.arange(n_queries, device=q.device)
            k_idx = torch.arange(n_keys, device=q.device)
            mask = q_idx[:, None] >= k_idx[None, :]
            scores = torch.where(mask, scores, torch.tensor(-1e6, device=q.device, dtype=scores.dtype))

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        lse = torch.logsumexp(scores, dim=-1)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        is_causal = ctx.is_causal

        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            q_idx = torch.arange(n_queries, device=q.device)
            k_idx = torch.arange(n_keys, device=q.device)
            mask = q_idx[:, None] >= k_idx[None, :]
            scores = torch.where(mask, scores, torch.tensor(-1e6, device=q.device, dtype=scores.dtype))

        probs = torch.exp(scores - lse.unsqueeze(-1))

        D = torch.sum(out * grad_out, dim=-1)

        grad_v = torch.matmul(probs.transpose(-2, -1), grad_out)
        grad_p = torch.matmul(grad_out, v.transpose(-2, -1))
        grad_s = probs * (grad_p - D.unsqueeze(-1))

        grad_q = torch.matmul(grad_s, k) * scale
        grad_k = torch.matmul(grad_s.transpose(-2, -1), q) * scale

        return grad_q, grad_k, grad_v, None


# Temporary fallback for Mac/local testing.
# The true Triton version will be added later on Koa/GCP.
FlashAttention2Triton = FlashAttention2PyTorch

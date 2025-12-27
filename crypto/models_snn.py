import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# SNN: surrogate spikes (From Reference)
# ----------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, thr: float):
        out = (u >= thr).to(u.dtype)
        # IMPORTANT: clone to avoid inplace-version issues later
        ctx.save_for_backward(u.clone())
        ctx.thr = thr
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (u,) = ctx.saved_tensors
        thr = ctx.thr
        # surrogate gradient (triangular around threshold)
        x = (u - thr).abs()
        grad = (x < 1.0).to(u.dtype) * (1.0 - x)
        return grad_out * grad, None

def spike(u: torch.Tensor, thr: float) -> torch.Tensor:
    return SpikeFn.apply(u, thr)

class SNNClassifier(nn.Module):
    """
    Simple LIF-like recurrent-in-time SNN with surrogate gradients.
    Adapted for Stock Data (Numerical features).
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int, steps: int = 20, beta: float = 0.9, thr: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True)
        self.steps = steps
        self.beta = beta
        self.thr = thr

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (Batch, Input_Dim) - Flattened feature vector for the window.
        Returns:
            logits: (Batch, Classes)
            mean_spikes: scalar (average spikes per neuron per step)
        """
        B = x.size(0)
        dev = x.device
        u1 = torch.zeros((B, self.fc1.out_features), device=dev)
        u2 = torch.zeros((B, self.fc2.out_features), device=dev)

        out_sum = torch.zeros((B, self.fc2.out_features), device=dev)
        spk_mean_accum = 0.0

        # compute input current once for speed (Static Input assumption)
        i1 = self.fc1(x)

        for _ in range(self.steps):
            u1 = self.beta * u1 + i1
            s1 = spike(u1, self.thr)
            # reset without inplace ops
            u1 = u1 * (1.0 - s1)

            u2 = self.beta * u2 + self.fc2(s1)
            s2 = spike(u2, self.thr)
            u2 = u2 * (1.0 - s2)

            out_sum = out_sum + s2
            spk_mean_accum += float(s1.sum(dim=1).mean().detach().cpu().item())

        logits = out_sum / float(self.steps)
        mean_spikes = spk_mean_accum / float(self.steps)
        return logits, mean_spikes

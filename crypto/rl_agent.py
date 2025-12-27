import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class GatePolicy(nn.Module):
    """
    Decides whether to trigger System 2 (Wake-on) based on System 1's state.
    Input features: [confidence, margin, entropy]
    """
    def __init__(self, in_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RLAgent:
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        lambda_energy: float = 0.05,
        oxytocin_bonus: float = 0.02
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = GatePolicy(in_dim=3, hidden=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.lambda_energy = lambda_energy
        self.oxytocin_bonus = oxytocin_bonus
        self.baseline = 0.0
        self.baseline_momentum = 0.9

    def get_features(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Extracts [confidence, margin, entropy] from SNN probabilities.
        """
        # confidence
        top2 = torch.topk(probs, k=2, dim=1).values
        conf = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=1)
        
        return torch.stack([conf, margin, entropy], dim=1)

    def train_step(
        self,
        snn_output_probs: torch.Tensor,
        snn_correct: torch.Tensor,       # Binary (B,) 1 if SNN is correct
        sys2_correct: torch.Tensor,      # Binary (B,) 1 if System 2 is correct
    ):
        """
        One step of REINFORCE training.
        """
        self.policy.train()
        feats = self.get_features(snn_output_probs).to(self.device)
        snn_correct = snn_correct.to(self.device)
        sys2_correct = sys2_correct.to(self.device)
        
        # 1. Forward pass: get p(wake)
        p_wake = self.policy(feats).squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        
        # 2. Sample action (wake or not)
        action = torch.bernoulli(p_wake) # 1: Wake, 0: Stay
        
        # 3. Calculate Reward
        # Reward = Correctness - EnergyPenalty + OxytocinBonus
        # If wake, final prediction is from System 2, else from SNN.
        is_correct = (action * sys2_correct) + ((1.0 - action) * snn_correct)
        
        reward = is_correct - (self.lambda_energy * action)
        reward += self.oxytocin_bonus * (is_correct * (1.0 - action))
        
        # 4. REINFORCE update with baseline
        r_mean = reward.mean().item()
        self.baseline = self.baseline_momentum * self.baseline + (1.0 - self.baseline_momentum) * r_mean
        advantage = reward - self.baseline
        
        # log_prob(a | p) = a * log(p) + (1-a) * log(1-p)
        log_prob = action * torch.log(p_wake) + (1.0 - action) * torch.log(1.0 - p_wake)
        loss = -(advantage.detach() * log_prob).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), r_mean, action.mean().item()

    def decide(self, snn_output_probs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Makes a wake decision.
        """
        self.policy.eval()
        with torch.no_grad():
            feats = self.get_features(snn_output_probs).to(self.device)
            p_wake = self.policy(feats).squeeze(-1)
            
            if deterministic:
                return p_wake >= 0.5
            else:
                return torch.bernoulli(p_wake).bool()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

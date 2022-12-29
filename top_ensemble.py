import torch
from torch import nn
from torch.nn import functional

from einops import reduce, rearrange
from einops.layers.torch import Reduce, Rearrange


class TopEnsemble(nn.Module):
    def __init__(self, input_dim, LayerConstructor, num_experts, constructor_args):
        super().__init__()
        self.experts = nn.ModuleList([LayerConstructor(*constructor_args) for _ in range(num_experts)])
        # self.scorers = Scorers(dim=dim, num_experts=num_experts)
        self.scorers = nn.Sequential(
            Reduce("b ... d -> b d", "mean"),
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),         # [b, n]
        )

    def forward(self, x):
        scores = self.scorers(x)
        indices = torch.argmax(scores, axis=-1)
        res = []
        for b, i in enumerate(indices):
            res.append(scores[b, i] * self.experts[i](x[b]))
        return rearrange(res, "... -> ...")

def LinearEnsemble(input_dim, output_dim, num_experts=16):
    return TopEnsemble(
        input_dim=input_dim,
        LayerConstructor=nn.Linear,
        num_experts=num_experts,
        constructor_args=(input_dim, output_dim)
    )

l = LinearEnsemble(12, 16)
x = torch.randn(1024, 12)
y = l(x)
y.sum().backward()

for name, param in l.named_parameters():
    if param.requires_grad:
        print(name, torch.norm(param.grad) if param.grad is not None else 0., sep="\t")
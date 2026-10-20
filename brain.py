"""A whole population of small neural networks, evaluated in one batched pass.

Each car has its own brain: a tiny network 13 -> 16 -> 16 -> 2 (tanh everywhere).
Instead of 100 separate network objects, ALL weights live in one tensor `genes` of
shape (P, G): row p is the full list of numbers ("genes") of car p's brain. The GA
works directly on these rows (copying, mixing and mutating them).

For the forward pass we view slices of each row as weight matrices:
    W1: (P, 13, 16)   b1: (P, 1, 16)
    W2: (P, 16, 16)   b2: (P, 1, 16)
    W3: (P, 16, 2)    b3: (P, 1, 2)
and use a *batched matrix multiply* (torch.bmm): it does P independent matrix
multiplications at once - car p's observation times car p's weights - so all 100
brains think in a single GPU/CPU call instead of 100 small ones.
"""
import torch


class PopulationBrain:
    def __init__(self, pop_size, layer_sizes, device="cpu", genes=None):
        self.sizes = tuple(layer_sizes)
        self.device = torch.device(device)
        self.n_genes = sum(i * o + o for i, o in zip(self.sizes[:-1], self.sizes[1:]))
        self.genes = genes if genes is not None else self.random_genes(pop_size)

    def random_genes(self, pop_size):
        """Random starting brains. Weights ~ N(0, 1/fan_in) keep tanh outputs in a useful range."""
        parts = []
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            parts.append(torch.randn(pop_size, i * o, device=self.device) / i ** 0.5)
            parts.append(torch.randn(pop_size, o, device=self.device) * 0.1)
        return torch.cat(parts, dim=1)

    def layers(self, genes):
        """Split each gene row into (W, b) per layer. These are views, no copying."""
        P = genes.shape[0]
        out, k = [], 0
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            W = genes[:, k:k + i * o].view(P, i, o)
            k += i * o
            b = genes[:, k:k + o].view(P, 1, o)
            k += o
            out.append((W, b))
        return out

    @torch.no_grad()
    def forward(self, obs, genes=None):
        """obs: (P, 13) -> actions (P, 2) in [-1, 1]. Car p uses brain p."""
        genes = self.genes if genes is None else genes
        x = obs.unsqueeze(1)                       # (P, 1, 13): each car's input as a 1-row matrix
        for W, b in self.layers(genes):
            # baddbmm = b + bmm(x, W): P matrix multiplies (1 x in) @ (in x out) in one call.
            x = torch.tanh(torch.baddbmm(b, x, W))
        return x.squeeze(1)

    __call__ = forward

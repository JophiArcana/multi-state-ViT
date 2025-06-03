import torch
import torch.nn as nn


if __name__ == "__main__":
    p = nn.Parameter(torch.randn(()))
    t = torch.atan(p)
    torch.autograd.grad(t, p)



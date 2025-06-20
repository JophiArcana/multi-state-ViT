import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from infrastructure import utils


if __name__ == "__main__":
    torch.manual_seed(1212)
    invc = utils.InverseCubic()
    t = 1.0 * torch.randn((1000000,))
    out = invc(t)

    plt.hist(out, bins=100)
    plt.show()



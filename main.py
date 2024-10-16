import numpy as np

from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

class  ViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches = 7):
        super(ViT, self).__init__()

        self.chw =chw   #channel width height
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape should be divisible by _patches"
        assert chw[2] % n_patches == 0, "Input shape should be divisible by _patches"

        self.patch_size = (chw[1]/ n_patches, chw[2] / n_patches)

        self.input_dim = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dim, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        #Positional Embedding
        self.pos_embed = nn.Parameter(torch.tensor(self.n_patches **2 + 1, self.hidden_d))
        self.pos_embed.requires_grad= False

    def forward(self, images):
        n, c, h, w = images
        patches = patch_embedding(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        tokens = torch.stack([torch.vstack(self.class_token, tokens[i])] for i in range(len(tokens)))

        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed
        return tokens



def patch_embedding(images, n_patches):

    n, c, h, w = images.shape # n, 1, 28, 28
    assert h == w, "Patch Embedding requires the h and w to be same"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = images[:, i * patch_size: (i + 1) * patch_size, j *patch_size: (j + 1)* patch_size]
                # image 2D patch 0 -------> 4
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches


def positional_embedding(sequence_length, d):

    result = torch.ones(sequence_length, d)

    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 ===0 else np.cos(i / (10000 ** (j -1)/d))

    return result



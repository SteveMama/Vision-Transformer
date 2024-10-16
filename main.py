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
    def __init__(self, chw=(1, 28, 28), n_patches = 7,n_heads = 2, n_blocks = 2, hidden_d =2, out_d = 10):
        super(ViT, self).__init__()

        self.chw =chw   #channel width height
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape should be divisible by _patches"
        assert chw[2] % n_patches == 0, "Input shape should be divisible by _patches"

        self.patch_size = (chw[1]/ n_patches, chw[2] / n_patches)

        self.input_dim = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dim, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        #Positional Embedding
        self.pos_embed = nn.Parameter(torch.tensor(self.n_patches **2 + 1, self.hidden_d))
        self.pos_embed.requires_grad= False


        self.blocks = nn.ModuleList(
            [
                EncoderVIT(self.hidden_d, self.n_heads) for _ in range(n_blocks)
            ]
        )

        # Classification MLP
        self.mlp  = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images
        patches = patch_embedding(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        tokens = torch.stack([torch.vstack(self.class_token, tokens[i])] for i in range(len(tokens)))

        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        #Transformer block
        for block in self.blocks:
            out = block(out)

        #Classification Token
        out = out[:, 0]


        return self.mlp(out)





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
             result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 ==0 else np.cos(i / (10000 ** (j -1)/d))

    return result


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f'Dimension {d} is not divisible by heads:{n_heads}'

        #patch --> q, k, v w.r.t attn , but not apply across n_heads
        d_head = int(d / n_heads)
        self.q = nn.ModuleList([nn.linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k = nn.ModuleList([nn.linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v = nn.ModuleList([nn.linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head =  d_head
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, sequences):
        # N, seq_leng, token_dim
        result = []
        for sequence in sequences:

            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q[head]
                k_mapping = self.k[head]
                v_mapping = self.k[head]

                seq = sequence[: head * self.d_head: (head+1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5)) # @ is dot product
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))


        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class EncoderVIT(nn.Module):

    def __init__(self, hidden_d, n_heads, mlp_ratio = 4):
        super(EncoderVIT, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads =  n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GeLU(),
            nn.Linear(mlp_ratio * hidden_d,hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


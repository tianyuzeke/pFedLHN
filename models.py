from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, n_embeds, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, 
            norm_var=0.01, layer_wise=True, use_fc=False, hdim=128):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_embeds, embedding_dim=embedding_dim)
        self.layer_wise = layer_wise
        # self.embeddings = nn.Parameter(torch.randn((n_embeds, embedding_dim)))

        self.use_fc = use_fc

        self.fc1 = nn.Linear(embedding_dim, hdim)

        num_params = 10

        self.c1_weights = nn.Linear(hdim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hdim, self.n_kernels)
        self.c2_weights = nn.Linear(hdim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hdim, 2 * self.n_kernels)
        if in_channels == 3:
            self.l1_weights = nn.Linear(hdim, 120 * 2 * self.n_kernels * 5 * 5)
        else:
            self.l1_weights = nn.Linear(hdim, 120 * 2 * self.n_kernels * 4 * 4)
        self.l1_bias = nn.Linear(hdim, 120)
        self.l2_weights = nn.Linear(hdim, 84 * 120)
        self.l2_bias = nn.Linear(hdim, 84)
        self.l3_weights = nn.Linear(hdim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hdim, self.out_dim)

        # self.coeff = nn.ParameterList([nn.Embedding(num_params, n_embeds) for i in range(n_nodes)])
        self.coeff = nn.ParameterList([nn.Parameter(torch.normal(0, norm_var, (num_params if layer_wise else 1, n_embeds))) \
            for i in range(n_nodes)])

    def forward(self, idx):
        # ft = torch.matmul(self.coeff[idx].weight, self.embeddings.weight)
        ft = torch.matmul(self.coeff[idx], self.embeddings.weight)
        if self.use_fc:
            weights = OrderedDict({
                "conv1.weight": self.c1_weights(self.fc1(ft[0])).view(self.n_kernels, self.in_channels, 5, 5),
                "conv1.bias": self.c1_bias(self.fc1(ft[1])).view(-1),
                "conv2.weight": self.c2_weights(self.fc1(ft[2])).view(2 * self.n_kernels, self.n_kernels, 5, 5),
                "conv2.bias": self.c2_bias(self.fc1(ft[3])).view(-1),
                "fc1.weight": self.l1_weights(self.fc1(ft[4])).view(120, 2 * self.n_kernels * 5 * 5) if self.in_channels==3 else self.l1_weights(self.fc1(ft[4])).view(120, 2 * self.n_kernels * 4 * 4),
                "fc1.bias": self.l1_bias(self.fc1(ft[5])).view(-1),
                "fc2.weight": self.l2_weights(self.fc1(ft[6])).view(84, 120),
                "fc2.bias": self.l2_bias(self.fc1(ft[7])).view(-1),
                "fc3.weight": self.l3_weights(self.fc1(ft[8])).view(self.out_dim, 84),
                "fc3.bias": self.l3_bias(self.fc1(ft[9])).view(-1),
            }) if self.layer_wise else OrderedDict({
                "conv1.weight": self.c1_weights(self.fc1(ft[0])).view(self.n_kernels, self.in_channels, 5, 5),
                "conv1.bias": self.c1_bias(self.fc1(ft[0])).view(-1),
                "conv2.weight": self.c2_weights(self.fc1(ft[0])).view(2 * self.n_kernels, self.n_kernels, 5, 5),
                "conv2.bias": self.c2_bias(self.fc1(ft[0])).view(-1),
                "fc1.weight": self.l1_weights(self.fc1(ft[0])).view(120, 2 * self.n_kernels * 5 * 5) if self.in_channels==3 else self.l1_weights(self.fc1(ft[0])).view(120, 2 * self.n_kernels * 4 * 4),
                "fc1.bias": self.l1_bias(self.fc1(ft[0])).view(-1),
                "fc2.weight": self.l2_weights(self.fc1(ft[0])).view(84, 120),
                "fc2.bias": self.l2_bias(self.fc1(ft[0])).view(-1),
                "fc3.weight": self.l3_weights(self.fc1(ft[0])).view(self.out_dim, 84),
                "fc3.bias": self.l3_bias(self.fc1(ft[0])).view(-1),
            }) 

        return weights
 

class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        if in_channels == 3:
            self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        else:
            self.fc1 = nn.Linear(2 * n_kernels * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channel_out=3, size=(28, 28)):
        super(Generator, self).__init__()
        self.channel_out = channel_out
        self.size = size

        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, channel_out*size[0]*size[1]),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gen(x)
        return out.view(x.size(0), self.channel_out, self.size[0], self.size[1])


class Discriminator(nn.Module):
    def __init__(self, channel_in=3, size=(28, 28)):
        super(Discriminator, self).__init__()
        self.channel_in = channel_in
        self.size = size

        self.disc = nn.Sequential(
            nn.Linear(self.channel_in*self.size[0]*self.size[1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.disc(x)
        return out
        






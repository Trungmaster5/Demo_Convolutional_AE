import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalAE(nn.Module):
    def __init__(self):
        super(ConvolutionalAE, self).__init__()

        # Encoder
        # Conv depth 3 --> 16 3x3 kernels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64, 7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride = 2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = self.decoder(x)
        return x
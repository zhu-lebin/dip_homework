import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Convolutional Layers with Residual Connections)
        self.enc1 = self.conv_block(3, 64, normalize=False)  # Downsample 1
        self.enc2 = self.conv_block(64, 128)                 # Downsample 2
        self.enc3 = self.conv_block(128, 256, dropout=True)  # Downsample 3 (Add Dropout)
        self.enc4 = self.conv_block(256, 512, dropout=True)  # Downsample 4 (Add Dropout)
        self.enc5 = self.conv_block(512, 512, dropout=True)  # Downsample 5 (Bottleneck, Add Dropout)

        # Decoder (Deconvolutional Layers with Skip Connections)
        self.dec1 = self.deconv_block(512, 512, dropout=True)  # Upsample 1
        self.dec2 = self.deconv_block(1024, 256, dropout=True) # Upsample 2
        self.dec3 = self.deconv_block(512, 128)                # Upsample 3
        self.dec4 = self.deconv_block(256, 64)                 # Upsample 4
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output layer for pixel generation
        )

    def conv_block(self, in_channels, out_channels, normalize=True, dropout=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.3))  # Add dropout to encoder
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))  # Add Dropout to prevent overfitting
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)  # Downsample 1
        e2 = self.enc2(e1) # Downsample 2
        e3 = self.enc3(e2) # Downsample 3
        e4 = self.enc4(e3) # Downsample 4
        e5 = self.enc5(e4) # Downsample 5

        # Decoder forward pass
        d1 = self.dec1(e5)              # Upsample 1
        d1 = torch.cat([d1, e4], dim=1) # Skip connection 1
        d2 = self.dec2(d1)              # Upsample 2
        d2 = torch.cat([d2, e3], dim=1) # Skip connection 2
        d3 = self.dec3(d2)              # Upsample 3
        d3 = torch.cat([d3, e2], dim=1) # Skip connection 3
        d4 = self.dec4(d3)              # Upsample 4
        d4 = torch.cat([d4, e1], dim=1) # Skip connection 4
        d5 = self.dec5(d4)              # Final output

        return d5

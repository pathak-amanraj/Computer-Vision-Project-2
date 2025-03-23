class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.ca = ChannelAttention(channels, reduction)

    def forward(self, x):
        out = self.body(x)
        out = self.ca(out)
        return x + out

class ResidualGroup(nn.Module):
    def __init__(self, channels, n_RCAB, reduction=16):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(channels, reduction) for _ in range(n_RCAB)]
        modules.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        return x + res

class RCAN(nn.Module):
    def __init__(self, scale_factor=4, n_resgroups=10, n_RCAB=20, channels=64, reduction=16):
        super(RCAN, self).__init__()
        # Shallow feature extraction
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        # Residual groups
        self.resgroups = nn.Sequential(*[ResidualGroup(channels, n_RCAB, reduction) for _ in range(n_resgroups)])
        # Conv after groups
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Upscaling using PixelShuffle (assuming scale_factor is power of 2)
        upscaling = []
        for _ in range(int(scale_factor//2)):
            upscaling += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upscale = nn.Sequential(*upscaling)
        # Reconstruction
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.resgroups(x)
        x = self.conv2(res) + x
        x = self.upscale(x)
        x = self.conv3(x)
        return x


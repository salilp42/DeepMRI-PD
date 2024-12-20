import torch
import torch.nn as nn
import torch.nn.functional as F

class BSpline3D(nn.Module):
    def __init__(self, num_channels, num_knots=10, grid_range=(-1, 1)):
        super(BSpline3D, self).__init__()
        self.num_channels = num_channels
        self.num_knots = num_knots
        self.grid_range = grid_range
        self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], num_knots))
        self.weights = nn.Parameter(torch.randn(num_channels, num_knots))

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        knots_expanded = self.knots.view(1, 1, 1, 1, 1, -1).expand(x.size(0), self.num_channels, x.size(2), x.size(3), x.size(4), self.num_knots)
        basis = F.relu(x_expanded - knots_expanded).pow(3)
        return torch.einsum('bcdhwk,ck->bcdhw', basis, self.weights)

class SplineConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SplineConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.spline = BSpline3D(out_channels)
        self.w1 = nn.Parameter(torch.randn(out_channels))
        self.w2 = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        conv_out = self.conv(x)
        spline_out = self.spline(conv_out)
        silu_out = F.silu(conv_out)
        return self.w1.view(1, -1, 1, 1, 1) * spline_out + self.w2.view(1, -1, 1, 1, 1) * silu_out

class ConvKAN3D(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(ConvKAN3D, self).__init__()
        self.conv1 = SplineConv3d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = SplineConv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = SplineConv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

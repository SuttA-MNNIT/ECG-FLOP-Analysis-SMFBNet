```python
import torch
import torch.nn as nn
from thop import profile, clever_format

class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        # Convolutional Block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        # Simplified Inception Block
        self.conv1x1 = nn.Conv1d(32, 16, kernel_size=1)
        self.conv3x3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.pool_path = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(32, 16, kernel_size=1)
        )
        # Bi-GRU
        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.bn(x)
        x = self.pool(x)
        # Inception: concatenate outputs
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.pool_path(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)  # 64 channels
        x = x.permute(0, 2, 1)  # For GRU: (batch, seq_len, features)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)  # Back to (batch, features, seq_len)
        x = self.bn_gru(x)
        x = self.dropout(x)
        return x


class ECGModel(nn.Module):
    def __init__(self, num_leads):
        super(ECGModel, self).__init__()
        self.branches = nn.ModuleList([BranchNet() for _ in range(num_leads)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * num_leads, 9)  # 9 classes

    def forward(self, x):
        # Process each lead through its branch
        branch_outputs = [branch(x[:, i:i+1, :]) for i, branch in enumerate(self.branches)]
        x = torch.cat(branch_outputs, dim=1)  # Concatenate along channel dim
        x = self.gap(x).squeeze(-1)  # Global Average Pooling
        x = self.fc(x)
        return x

def compute_flops(num_leads):
    model = ECGModel(num_leads)
    input_tensor = torch.randn(1, num_leads, 7500)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params

baseline_flops, baseline_params = compute_flops(12)
smfb_flops, smfb_params = compute_flops(9)  # Approximate 9.8 as 9 for simplicity


baseline_flops, baseline_params = clever_format([baseline_flops, baseline_params], "%.3f")
smfb_flops, smfb_params = clever_format([smfb_flops, smfb_params], "%.3f")

print(f"Baseline (12 leads): FLOPs = {baseline_flops}, Params = {baseline_params}")
print(f"SMFB-Net (9 leads): FLOPs = {smfb_flops}, Params = {smfb_params}")
reduction = (1 - float(smfb_flops.split()[0]) / float(baseline_flops.split()[0])) * 100
print(f"Computational Reduction: {reduction:.2f}%")
```
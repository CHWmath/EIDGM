import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, b_input, t_input, output_dim, b_layers=3, t_layers=3, width=128):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(b_input, width),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(width, width), nn.Tanh()) for _ in range(b_layers-1)]
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(t_input, width),
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(width, width), nn.Tanh()) for _ in range(t_layers-1)]
        )
        self.final_linear = nn.Linear(2*width, 1)
        
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = (branch_output * trunk_output).sum(dim=1, keepdim=True)
        return output
import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=4):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim //2, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        
        self.action_dim = action_dim
        assert action_dim == 4
        self.norm_factor = 50
        self.state_dim = state_dim

    def forward(self, x):
        if x.ndim == 1:
            squeeze = True
        else:
            squeeze = False
        x = self.expand_inputs_to_action_dim(x)
        x = self.normalize_data(x)
        x = self.transform_state_to_diff(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output(x)
        if squeeze:
            return x.flatten()
        return x.squeeze(2)
    
    def expand_inputs_to_action_dim(self, x):

        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2

        batch_size = x.shape[0]
        state_dim = x.shape[1]

        xright = x.clone()
        xright[:, 0] = x[:, 0] + 1
        xright = xright.unsqueeze(1)
        xup = x.clone()
        xup[:, 1] = x[:, 1] + 1
        xup = xup.unsqueeze(1)
        xleft = x.clone()
        xleft[:, 0] = x[:, 0] - 1
        xleft = xleft.unsqueeze(1)
        xdown = x.clone()
        xdown[:, 1] = x[:, 1] - 1
        xdown = xdown.unsqueeze(1)
        x_expanded = torch.concat([xright, xup, xleft, xdown], dim=1)

        assert x_expanded.shape == (batch_size, self.action_dim, state_dim)
        return x_expanded
    
    def normalize_data(self, data):
        norm_data = (data - self.norm_factor) / self.norm_factor
        return norm_data
    
    def transform_state_to_diff(self, x):
        n = self.state_dim // 2 
        diff = x[:,:,:n] - x[:,:,n:]
        return diff


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)

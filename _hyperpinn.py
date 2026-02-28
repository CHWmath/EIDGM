import torch
import torch.nn as nn

class DynamicFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    def forward(self, x, weights):
        batch_size = x.size(0)
        weight_matrix = weights[:, :self.input_dim * self.output_dim].reshape(batch_size, self.output_dim, self.input_dim)
        bias = weights[:, self.input_dim * self.output_dim:].reshape(batch_size, self.output_dim)
        return torch.baddbmm(bias.unsqueeze(1), x.unsqueeze(1), weight_matrix.transpose(1, 2)).squeeze(1)    
    
class TrunkNetwork(nn.Module):
    def __init__(self, layers_config, act=nn.Tanh()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        for input_dim, output_dim in layers_config[:-1]: 
            self.layers.append(DynamicFC(input_dim, output_dim))
            self.acts.append(act) 
        self.layers.append(DynamicFC(*layers_config[-1]))
    def forward(self, x, params):
        for layer, act, param in zip(self.layers[:-1], self.acts, params[:-1]):
            x = layer(x, param)
            x = act(x)
        x = self.layers[-1](x, params[-1])
        return x

class HyperNetwork(nn.Module):
    def __init__(self, num_sensors, num_chunks_in, total_params, depth, width, activation):
        super().__init__()
        self.num_chunks_in = num_chunks_in
        self.latent_chunk = nn.Parameter(torch.randn(num_chunks_in))
        activations = {
            'tanh': nn.Tanh(),
            'prelu': nn.PReLU(),
            'relu': nn.ReLU()
        }
        act = activations.get(activation, None)
        if act is None:
            raise ValueError("Unsupported activation type. Choose from 'tanh', 'prelu', 'relu'.")

        layers = [nn.Linear(num_sensors + num_chunks_in, width), act]
        for _ in range(1, depth):
            layers.extend([nn.Linear(width, width), act])
        layers.append(nn.Linear(width, total_params))
        self.network = nn.Sequential(*layers)
    def forward(self, sensor_data):
        batch_size = sensor_data.size(0)
        repeated_chunk = self.latent_chunk.repeat(batch_size, 1)
        hyper_input = torch.cat((sensor_data, repeated_chunk), dim=1)
        return self.network(hyper_input)    

class HyperNetwork_lorenz(nn.Module):
    def __init__(self, num_sensors, total_params, depth, width, activation):
        super().__init__()
        activations = {
            'tanh': nn.Tanh(),
            'prelu': nn.PReLU(),
            'relu': nn.ReLU()
        }
        act = activations.get(activation, None)
        if act is None:
            raise ValueError("Unsupported activation type. Choose from 'tanh', 'prelu', 'relu'.")
        layers = [nn.Linear(num_sensors, width), act]
        for _ in range(1, depth):
            layers.extend([nn.Linear(width, width), act])
        layers.append(nn.Linear(width, total_params))
        self.network = nn.Sequential(*layers)
    def forward(self, sensor_data):
        return self.network(sensor_data)      
    
def compute_total_params(layers_config):
    total = sum((inp * out + out) for inp, out in layers_config)
    return total
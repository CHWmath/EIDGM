import torch.nn as nn
import torch.nn.functional as F

class G_model(nn.Module) :
    def __init__(self, hidden_dims) :                    # Hidden_dims : [h1, h2, h3, ..., hn]
        super(G_model, self).__init__()
        self.layers = []
        for i in range(len(hidden_dims)-1) :
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1])) # hidden layers
        self.layers = nn.ModuleList(self.layers)
        for layer in self.layers :                       # Weight initialization
            nn.init.xavier_uniform_(layer.weight)        # Also known as Glorot initialization
        self.act = nn.LeakyReLU(0.2, inplace=True)      # Nonlinear activation function
        self.act1 = nn.Tanh()
    def forward(self, x) :
        x = self.act1(self.layers[0](x))
        for layer in self.layers[1:-2] :
            x = self.act1(layer(x))
        x = self.layers[-1](self.act1(self.layers[-2](x)))
        return x
        
class D_model(nn.Module) :
    def __init__(self, hidden_dims) :                    # Hidden_dims : [h1, h2, h3, ..., hn]
        super(D_model, self).__init__()
        self.layers = []
        for i in range(len(hidden_dims)-1) :
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1])) # hidden layers
        self.layers = nn.ModuleList(self.layers)
        for layer in self.layers :                       # Weight initialization
            nn.init.xavier_uniform_(layer.weight)        # Also known as Glorot initialization
        #self.act = nn.LeakyReLU(0.2, inplace=True)      # Nonlinear activation function
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
    def forward(self, x) :
        x = self.act2(self.layers[0](x))
        for layer in self.layers[1:-1] :
            x = self.act2(layer(x))
        x = self.layers[-1](x)
        return x
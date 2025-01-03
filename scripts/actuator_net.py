import torch 
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, act, slope=0.05):
        super(Activation, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input):
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1.)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1.)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1.) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1.) -
                self.slope * F.relu(-input) -
                self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        elif self.act == "softsign":
            return F.softsign(input)
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")

class ActuatorNet(nn.Module):
    def __init__(self, input_size=6, output_size=1, hidden_size=32, layer_num=3, activation='softsign'):
        super(ActuatorNet, self).__init__()
        
        layers = []
        for i in range(layer_num):
            if i == 0:
                # First layer
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(Activation(activation))
            elif i == layer_num - 1:
                # Last layer
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(Activation(activation))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    

if __name__ == '__main__':
    net = ActuatorNet()
    print(net)
    x = torch.randn(32, 6)
    y = net(x)
    print(y.shape)
    print(y)
    print(y.mean())
    print(y.std())
    print(y.max())
    print(y.min())
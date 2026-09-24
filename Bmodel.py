import torch.nn as nn
import numpy as np
from Aconfig import Config, DEVICE
import torch

config = Config()

class PINN1(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Input layer

        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers

        for _ in range(config.num_hidden_layers -1 ):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.Tanh())


        # output layer

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)


    def forward(self, x, t):
        inputs = torch.cat((x,t), dim= 1)

        return self.network(inputs)

    
class PINN2(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Input layer

        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers

        for _ in range(config.num_hidden_layers -1 ):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.Tanh())


        # output layer

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)


    def forward(self, x, t):
        inputs = torch.cat((x,t), dim= 1)

        return self.network(inputs)

    
class PINN3(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Input layer

        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers

        for _ in range(config.num_hidden_layers -1 ):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.Tanh())


        # output layer

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)


    def forward(self, x, t):
        inputs = torch.cat((x,t), dim= 1)

        return self.network(inputs)

    
class PINN4(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Input layer

        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers

        for _ in range(config.num_hidden_layers -1 ):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.Tanh())


        # output layer

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)


    def forward(self, x, t):
        inputs = torch.cat((x,t), dim= 1)

        return self.network(inputs)

    



for param in PINN1().parameters():
    print(param)
    

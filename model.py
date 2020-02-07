import numpy as np
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class SubNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.layers.extend([nn.Linear(i, o) for i, o in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.reset_parameters()

    def forward(self, x, action=None):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

    def reset_parameters(self):
        for l in self.layers[:-1]:
            l.weight.data.uniform_(*hidden_init(l))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

class ICM(nn.Module):
    
    def __init__(
        self,
        state_size=None,
        action_size=None,
        hidden_layers=[256, 128],
        lr=1e-3,
        n=0.01,
        beta=0.1
    ):
        super().__init__()
        feature_size = hidden_layers[0]

        self.beta = beta
        self.n = n
        self.bn = nn.BatchNorm1d(state_size, momentum=0.1)
        self.phi = nn.Linear(state_size, feature_size)
        self.phiI = SubNetwork(feature_size + feature_size, action_size, hidden_layers[1:])
        self.phiF = SubNetwork(feature_size + action_size, feature_size, hidden_layers[1:])
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action, next_state):
        x = F.relu(self.phi(self.bn(state)))
        x = torch.cat((x, action), dim=1)
        predicted_next_state = self.phiF(x)
        next_state = self.phi(next_state)
        return 0.5 * (next_state - predicted_next_state).pow(2).mean(1).unsqueeze(1)

    def inverse(self, state, action, next_state):
        x1 = F.relu(self.phi(self.bn(state)))
        x2 = F.relu(self.phi(self.bn(next_state)))
        x = torch.cat((x1, x2), dim=1)
        predicted_action = self.phiI(x)
        return F.mse_loss(action, predicted_action)

    def surprise(self, state, action, next_state):
        Lf = self.forward(state, action, next_state)
        Li = self.inverse(state, action, next_state)
        self.learn(Lf.mean(), Li)
        return self.n * Lf.detach()

    def learn(self, Lf, Li):
        loss = (1-self.beta) * Li + self.beta * Lf
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()

class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
        super().__init__()
        self.network = SubNetwork(state_size, action_size, hidden_layers)

    def forward(self, state):
        x = self.network(state)
        return torch.tanh(x)

class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(state_size, momentum=0.1)
        self.phi = nn.Linear(state_size, hidden_layers[0])        
        self.network = SubNetwork(hidden_layers[0] + action_size, 1, hidden_layers[1:])
        self.reset_parameters()

    def forward(self, state, action):
        x = F.relu(self.phi(self.bn(state)))
        x = torch.cat((x, action), dim=1)
        return self.network(x)

    def reset_parameters(self):
        self.phi.weight.data.uniform_(*hidden_init(self.phi))


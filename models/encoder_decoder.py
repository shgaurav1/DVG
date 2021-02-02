from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 40)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h2 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h2))


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(90, 50)
        self.fc2 = nn.Linear(50, 6)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     x = F.dropout(x, training=self.training)
    #     return F.tanh(x)

    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        # h1 = F.dropout(h1, training=self.training)
        h2 = self.fc2(h1)
        return h2


class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 6)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     x = F.dropout(x, training=self.training)
    #     return F.tanh(x)

    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        # h1 = F.dropout(h1, training=self.training)
        h2 = self.fc2(h1)
        return h2
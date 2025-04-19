import torch
import torch.nn as nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_items, n_embd=16):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, n_embd)
        self.item_embedding = nn.Embedding(num_items, n_embd)
        
        self.fc1 = nn.Linear(n_embd * 2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
       
        self.sigmoid = nn.Sigmoid()
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        
        x = torch.cat([user_embed, item_embed], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        x = x * 5
        return x 
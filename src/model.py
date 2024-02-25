import torch
import torch.nn.functional as F

class LinearRegression(torch.nn.Module):
    def __init__(self, m=0.0, c=0.5):
        super().__init__()
        self.m = torch.nn.parameter.Parameter(torch.tensor([[m]], dtype=torch.float32))
        self.c = torch.nn.parameter.Parameter(torch.tensor([[c]], dtype=torch.float32))
    
    def forward(self, x):
        x = torch.matmul(x, self.m)
        x = x + self.c
        return x

class LogisticRegression(torch.nn.Module):
    def __init__(self, m=0.0, c=0.5):
        super().__init__()
        self.m = torch.nn.parameter.Parameter(torch.tensor([[m]], dtype=torch.float32))
        self.c = torch.nn.parameter.Parameter(torch.tensor([[c]], dtype=torch.float32))
    
    def forward(self, x):
        x = torch.matmul(x, self.m)
        x = x + self.c
        x = F.sigmoid(x)
        return x
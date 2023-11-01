import torch.nn as nn

class  BinaryClassifier(nn.Module):
    def __init__(self, num_features=171):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.Dropout(x)
        x = self.fc2(x)  
        return x

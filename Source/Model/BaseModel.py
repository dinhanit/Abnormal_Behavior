import torch.nn as nn

class  BinaryClassifier(nn.Module):
    def __init__(self, num_features=174):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer after the first fully connected layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer after the second fully connected layer
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(p=0.5)  # Dropout layer after the third fully connected layer
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)  # Apply dropout
        x = self.fc4(x)  # Output layer
        return x
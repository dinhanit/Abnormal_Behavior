import torch.nn as nn
import torch


#4fc
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



#3fc
class  BinaryClassifier_2(nn.Module):
    def __init__(self, num_features=174):
        super(BinaryClassifier_2, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer after the first fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer after the second fully connected layer
        self.fc3 = nn.Linear(64, 2)

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
        return x


#5fc
class BinaryClassifier_3(nn.Module):
    def __init__(self, num_features=174):
        super(BinaryClassifier_3, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer after the first fully connected layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer after the second fully connected layer
        self.fc3 = nn.Linear(128, 64)  # Add a third fully connected layer
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(p=0.5)  # Dropout layer after the third fully connected layer
        self.fc4 = nn.Linear(64, 32)  # Add a fourth fully connected layer
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(p=0.5)  # Dropout layer after the fourth fully connected layer
        self.fc5 = nn.Linear(32, 2)  # Add a fifth fully connected layer

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
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)  # Apply dropout
        x = self.fc5(x)  # Output layer
        return x









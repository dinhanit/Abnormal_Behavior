from BaseModel import BinaryClassifier
import torch.nn as nn
import torch
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from param import *
    
#Load DataSet
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)  # Convert to PyTorch tensor
        self.labels = torch.LongTensor(labels)  # Assuming labels are 0 or 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:',DEVICE)
model = BinaryClassifier()
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)
criterion.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)

data_train = np.load("CombinedData_train.npz")
x = data_train['landmarks']
y = data_train['labels']
dataset_train = CustomDataset(x, y)

data_train = np.load("CombinedData_train.npz")
data_train = CustomDataset(data_train['landmarks'],data_train['labels'])

data_test = np.load("CombinedData_test.npz")
data_test = CustomDataset(data_test['landmarks'],data_test['labels'])

TRAINLOADER = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)



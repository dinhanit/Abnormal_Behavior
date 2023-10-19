from BaseModel import BinaryClassifier
import torch.nn as nn
import torch
import torch.optim as optim
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

print('DEVICE:',DEVICE)
model = BinaryClassifier()
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)
criterion.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)


data_train = np.load("Data/CombinedData_train.npz")
data_train = CustomDataset(data_train['landmarks'],data_train['labels'])

data_test = np.load("Data/CombinedData_test.npz")
data_test = CustomDataset(data_test['landmarks'],data_test['labels'])

TRAINLOADER = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)



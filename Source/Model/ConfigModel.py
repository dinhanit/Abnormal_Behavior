from BaseModel import *
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from param import *
from torchsummary import summary
    
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
model = BinaryClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)
criterion.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
# optimizer = optim.SGD(model.parameters(), lr=float(LEARNING_RATE))

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)


data_train = np.load("../Data/Preprocessed_Data_train.npz")
data_train = CustomDataset(data_train['landmarks'],data_train['labels'])

data_test = np.load("../Data/Preprocessed_Data_test.npz")
data_test = CustomDataset(data_test['landmarks'],data_test['labels'])

TRAINLOADER = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)


summary(model,(1,1,136))

from BaseModel import CustomCNN
import torch.nn as nn
import torch
import torch.optim as optim
import json
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

# Load hyperparameter
with open('param.json', 'r') as json_file:
    data = json.load(json_file)
for key, value in data.items():
    globals()[key] = value
    
#Load DataSet
def LoadData(name=""):
    data = np.load(name+".npz")
    images = data['images']
    labels = data['labels']

    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)

    dataset = TensorDataset(images, labels)
    return dataset

#Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:',DEVICE)
model = CustomCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)
criterion.to(DEVICE)
batch_size = BATCH_SIZE

optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)

TRAINLOADER = DataLoader(LoadData('Data_train'), batch_size=batch_size, shuffle=True)
TESTLOADER = DataLoader(LoadData('Data_test'), batch_size=batch_size, shuffle=False)



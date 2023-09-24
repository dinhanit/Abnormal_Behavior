from BaseModel import CustomCNN
from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import torch.optim as optim
import json

import json

# Open and read the JSON file
with open('Model/param.json', 'r') as json_file:
    data = json.load(json_file)

for key, value in data.items():
    globals()[key] = value

TRAINLOADER,TESTLOADER = None,None

model = CustomCNN()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=pram['Learning_rate'])
optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)


loss_train = []
loss_test = []
f1_train = []
f1_test = []



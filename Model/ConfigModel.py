from BaseModel import CustomCNN
from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import torchvision

with open('Model/param.json', 'r') as json_file:
    data = json.load(json_file)

for key, value in data.items():
    globals()[key] = value
    
train_transforms = transforms.Compose([
    transforms.Resize((224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 64
train_data = torchvision.datasets.ImageFolder(root='./Data/DataSets/SplitData/test', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root='./Data/DataSets/SplitData/train', transform=test_transforms)

TRAINLOADER = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = CustomCNN(num_classes=2)

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



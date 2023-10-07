import torchvision
from torchvision import transforms
import pickle

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

train_data = torchvision.datasets.ImageFolder(root='./DataSets/SplitData/test', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root='./DataSets/SplitData/train', transform=test_transforms)


with open('train_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)

with open('test_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)
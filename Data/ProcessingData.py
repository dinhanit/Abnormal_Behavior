import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import pickle

# Chuẩn bị transform
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

train_data = torchvision.datasets.ImageFolder(root='./Data/DataSets/SplitData/train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root='./Data/DataSets/SplitData/test', transform=test_transforms)
# Số lượng các lớp
num_classes = len(train_data.classes)
# Tên của các lớp
classes_name = train_data.classes
# classes2idx: ánh xạ từ tên lớp sang chỉ số (index)
# classes2idx = train_data.class_to_idx
# print(f"Số lượng lớp: {num_classes}")
# print(f"Tên lớp: {classes_name}")
# print(f"Ánh xạ từ tên lớp sang chỉ số: {classes2idx}")
# print("Number of train: ", len(train_data))
# print("Number of test: ", len(test_data))

with open('train_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)

with open('test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)



def show_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt
   
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    img_transformed = train_transforms(img)
    img_transformed = img_transformed.numpy().transpose(1,2,0) #chuyển định dạng và thứ tự size, channels
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()
# show_image('k.jpg')



def Data_Loader(data, batch_size):
    # Tạo DataLoader
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Kiểm tra kích thước của trainset và testset
    for batch in data_loader:
        batch_shape = len(batch[0])
        break  # Exit after checking the first batch
    print(f"Shape of the data batches: {batch_shape}")

    print('Load Done')
    return data_loader

# batch_size = 64 
# # Get X and y for train_data and save as .pth files
# torch.save(Data_Loader(train_data, batch_size), 'train_data.pth')

# # Get X and y for test_data and save as .pth files
# torch.save(Data_Loader(test_data, batch_size), 'test_data.pth')
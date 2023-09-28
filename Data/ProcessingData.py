import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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

# Tải dữ liệu
train_data = torchvision.datasets.ImageFolder(root='./DataSets/SplitData/train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root='./DataSets/SplitData/test', transform=test_transforms)
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

def Data_Loader(data):
    # Tạo DataLoader
    data_loader = DataLoader(data, batch_size=len(data), shuffle=True)

    # Lặp qua DataLoader để lấy dữ liệu và nhãn
    for images, labels in data_loader:
        X = images
        y = labels
        break  # Dừng sau một lần lặp để tránh lặp tiếp

    # Kiểm tra kích thước của X_train và y_train
    print(f"Kích thước X:", X.shape)  # Kích thước X_train: (số lượng mẫu, số kênh, chiều cao, chiều rộng)
    print(f"Kích thước y:", y.shape)  # Kích thước y_train: (số lượng mẫu,)
    return X, y

Data_Loader(train_data)
Data_Loader(test_data)


def show_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    img_transformed = train_transforms(img)
    img_transformed = img_transformed.numpy().transpose(1,2,0) #chuyển định dạng và thứ tự size, channels
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()
# show_image('k.jpg')
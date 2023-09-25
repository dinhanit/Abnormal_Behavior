import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))  
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(128 * 7 * 7, 128)) 
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 10)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()

        # Define the number of channels and layers for each block
        num_channels = [64, 128, 256, 512, 512]
        num_layers = [2, 2, 4, 4, 4]

        self.features = self._make_layers(num_channels, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, num_channels, num_layers):
        layers = []
        in_channels = 3  # Initial input channels

        for i, (out_channels, layers_count) in enumerate(zip(num_channels, num_layers)):
            # Add convolutional layers
            for j in range(layers_count):
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
                in_channels = out_channels

            # Add max-pooling layer after each block
            if i < len(num_channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of the VGG19 model

# model = VGG19()


# Define the ResNet class
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        
        # If the input and output dimensions do not match, use a convolutional layer to match them
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# resnet = ResNet(BasicBlock, [2, 2])
# print(resnet)

#FaceNet
class FaceNetBinary(nn.Module):
    def __init__(self, embedding_size):
        super(FaceNetBinary, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, embedding_size)
        self.fc3 = nn.Linear(embedding_size, 2)  # Output for binary classification
        
        # Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Dropout layer (optional)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalization of the embeddings
        x = nn.functional.normalize(x, p=2, dim=1)
        
        # Classification layer for binary classification
        x = self.fc3(x)
        
        return x
    
class EfficientNetLike(nn.Module):
    def __init__(self, num_classes=2, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNetLike, self).__init__()

        # Define the scaling factors for width and depth
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient

        # Define the number of channels for each block
        num_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]

        # Define the number of layers for each block
        num_layers = [1, 2, 2, 3, 3, 4, 1]

        # Initial Convolution Layer
        self.features = [nn.Conv2d(3, int(32 * width_coefficient), 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(int(32 * width_coefficient)),
                         nn.ReLU6(inplace=True)]

        # Building the model blocks
        for i in range(7):
            for j in range(num_layers[i]):
                self.features.append(self._build_block(num_channels[i], num_channels[i+1], 6, dropout_rate))

        self.features = nn.Sequential(*self.features)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(1280 * width_coefficient), num_classes)
        )

    def _build_block(self, in_channels, out_channels, expand_ratio, dropout_rate):
        layers = []
        if expand_ratio != 1:
            # Pointwise Convolution
            layers.append(nn.Conv2d(in_channels, int(in_channels * expand_ratio), 1, bias=False))
            layers.append(nn.BatchNorm2d(int(in_channels * expand_ratio)))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise Convolution
        layers.append(nn.Conv2d(int(in_channels * expand_ratio), int(in_channels * expand_ratio), 3, stride=1, padding=1, groups=int(in_channels * expand_ratio), bias=False))
        layers.append(nn.BatchNorm2d(int(in_channels * expand_ratio)))
        layers.append(nn.ReLU6(inplace=True))

        # Pointwise Convolution
        layers.append(nn.Conv2d(int(in_channels * expand_ratio), out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



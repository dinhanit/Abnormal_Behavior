import torch.nn as nn
class CustomCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(CustomCNN, self).__init__()
        self.sq1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.sq2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.sq3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        
        self.sq_classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.sq1(x)
        x = self.sq2(x)
        x = self.sq3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.sq_classifier(x)
        return x
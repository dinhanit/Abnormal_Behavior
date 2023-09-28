import torch.nn as nn

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
        self.features = self._make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._make_classifier(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_features(self):
        layers = []
        in_channels = 3
        # Configuration for VGG19
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v

        return nn.Sequential(*layers)


    def _make_classifier(self, num_classes):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

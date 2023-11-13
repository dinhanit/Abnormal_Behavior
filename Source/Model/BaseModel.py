from torch import nn

class BinaryClassifier(nn.Module):
    """
    Binary Classifier Module.

    Args:
        num_features (int): Number of input features.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout1d): Dropout layer.
        fc2 (nn.Linear): Second fully connected layer.
    """
    def __init__(self, num_features=171):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  
        return x

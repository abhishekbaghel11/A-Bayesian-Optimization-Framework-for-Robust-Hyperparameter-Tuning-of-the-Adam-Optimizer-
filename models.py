import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self, dataset_name="MNIST"):
        super(CNNNetwork, self).__init__()

        # Conditions for the dataset to be used
        dataset_upper = dataset_name.upper()
        if dataset_upper == "MNIST":
            in_channels = 1 
            num_classes = 10
        elif dataset_upper in ["CIFAR10", "IMBALANCED_CIFAR10"]:
            in_channels = 3
            num_classes = 10
        elif dataset_upper == "CIFAR100":
            in_channels = 3 
            num_classes = 100
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(3, 3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 48)  # Penultimate layer
        self.fc3 = nn.Linear(48, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, return_features=False):
        x = F.elu(self.conv1(x))
        x = self.pool(x)
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        x = F.elu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        features = self.fc2(x) 
        x = self.dropout(features)
        x = self.fc3(x)
        
        if return_features:
            return x, features
        return x
import torch
import torch.nn as nn
import torch.optim as optim

def create_cnn_model(input_shape, num_classes=2):
    
    channels = input_shape[2]  
    height = input_shape[0]
    width = input_shape[1]

    
    class CNNModel(nn.Module):
        def __init__(self, channels, num_classes):
            super(CNNModel, self).__init__()
            
            # First Convolutional Layer
            self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)  
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
            
            # Second Convolutional Layer
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
            
            # Third Convolutional Layer
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  
            
           
            
            self.flat_size = (height // 8) * (width // 8) * 128
            
            
            self.flatten = nn.Flatten()
            
            # Dense Layer 1
            self.fc1 = nn.Linear(self.flat_size, 128)
            self.relu1 = nn.ReLU()
            
            # Dropout Layer
            self.dropout = nn.Dropout(0.5)
            
            # Dense Layer 2
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            
            # Output Layer
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            # Pass through Conv1, Pool1, and ReLU
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.relu1(x)
            
            # Pass through Conv2, Pool2, and ReLU
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.relu1(x)
            
            # Pass through Conv3, Pool3, and ReLU
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.relu1(x)
            
            # Flatten the output
            x = self.flatten(x)
            
            # Fully connected layers
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout(x)
            
            x = self.fc2(x)
            x = self.relu2(x)
            
            x = self.fc3(x)  
            return x
    
    
    model = CNNModel(channels, num_classes)

    return model


input_shape = (64, 64, 3) 
num_classes = 2  
model = create_cnn_model(input_shape, num_classes)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

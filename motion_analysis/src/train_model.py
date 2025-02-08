import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np



X_train = np.load('D:/minor_project/motion_analysis/data/training/X_train.npy')
y_train = np.load('D:/minor_project/motion_analysis/data/training/y_train.npy')
X_test = np.load('D:/minor_project/motion_analysis/data/testing/X_test.npy')
y_test = np.load('D:/minor_project/motion_analysis/data/testing/y_test.npy')


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


model = create_cnn_model(input_shape=(64, 64, 3), num_classes=num_classes)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


epochs = 35
batch_size = 32
num_batches = len(X_train_tensor) // batch_size

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Shuffle the training data 
    permutation = torch.randperm(X_train_tensor.size(0))
    
    for i in range(0, len(X_train_tensor), batch_size):
        # Get mini-batch
        indices = permutation[i:i + batch_size]
        batch_inputs, batch_labels = X_train_tensor[indices], y_train_tensor[indices]
        
        # clear the old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    
    epoch_accuracy = (correct_predictions / total_samples) * 100
    print(f"Accuracy: {epoch_accuracy:.2f}%")

# Save the trained model
model_save_path = "D:/minor_project/motion_analysis/models/motion_detection/motion_detection_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# Load the model 
model = create_cnn_model(input_shape=(64, 64, 3), num_classes=num_classes)
model.load_state_dict(torch.load(model_save_path))
model.eval()  


with torch.no_grad():  # Disable gradient calculation
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    correct_predictions = (predicted == y_test_tensor).sum().item()
    test_accuracy = correct_predictions / len(y_test_tensor) * 100

print(f"Test Accuracy: {test_accuracy:.2f}%")

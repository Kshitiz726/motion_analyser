import numpy as np
from cnn_model import create_cnn_model
from tensorflow.keras.models import load_model

# Load the training and test data
X_train = np.load('D:/minor_project/motion_analysis/data/training/X_train.npy')
y_train = np.load('D:/minor_project/motion_analysis/data/training/y_train.npy')
X_test = np.load('D:/minor_project/motion_analysis/data/testing/X_test.npy')
y_test = np.load('D:/minor_project/motion_analysis/data/testing/y_test.npy')

# Normalize the frames to [0, 1] for CNN input
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define input shape based on the frame shape (e.g., 64x64x3 if you resized your frames to 64x64)
input_shape = X_train.shape[1:]

# Create the CNN model
num_classes = len(np.unique(y_train))  # Number of unique classes in y_train
model = create_cnn_model(input_shape, num_classes)

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=1
)

# Save the model
model_save_path = "D:/minor_project/motion_analysis/models/motion_detection/motion_detection_model.h5"
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Load the saved model (optional, to confirm saving works correctly)
model = load_model(model_save_path)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Print only the accuracy
print(f"Test Accuracy: {test_accuracy * 100:.2f} %")


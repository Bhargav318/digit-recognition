# Handwritten Digit Recognition using Neural Networks
# Dataset: MNIST

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Step 2: Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 3: Build Neural Network Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # Convert 2D image to 1D
    keras.layers.Dense(128, activation='relu'),   # Hidden layer
    keras.layers.Dense(64, activation='relu'),    # Hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer (digits 0-9)
])

# Step 4: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
print("Training the model...")
model.fit(x_train, y_train, epochs=5)

# Step 6: Evaluate the Model
print("\nTesting the model...")
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)

# Step 7: Make Predictions
predictions = model.predict(x_test)

# Step 8: Show Prediction Example
index = 0
print("\nPredicted Digit:", np.argmax(predictions[index]))
print("Actual Digit:", y_test[index])

# Step 9: Display Image
plt.imshow(x_test[index], cmap='gray')
plt.title("Test Image")
plt.show()
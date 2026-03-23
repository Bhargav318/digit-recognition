from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from utils import show_predictions
import os

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Check if model already exists
if os.path.exists("digit_model.h5"):
    print("Loading saved model...")
    model = keras.models.load_model("digit_model.h5")
else:
    print("Training model...")
    model = create_model()
    history = model.fit(x_train, y_train, epochs=5)

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # Save model
    model.save("digit_model.h5")

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Predict
predictions = model.predict(x_test)

# Show results
show_predictions(x_test, y_test, predictions)
import matplotlib.pyplot as plt
import numpy as np

def show_predictions(x_test, y_test, predictions):
    for i in range(5):
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
        plt.show()
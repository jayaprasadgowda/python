import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Regression Class
class KNNRegressor:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            # Calculate distances from x_test to all points in the training set
            distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]

            # Get the indices of the K nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get the target values of the K nearest neighbors
            k_nearest_values = self.y_train[k_indices]

            # Predict by taking the average of the K nearest neighbors' target values
            prediction = np.mean(k_nearest_values)
            predictions.append(prediction)

        return np.array(predictions)

# Example dataset for regression (Simple linear relationship with some noise)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Feature (X)
y = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.1, 4.6, 5.1, 5.7, 6.0])  # Target (y)

# Train/test split (simple, for demonstration purposes)
X_train = X[:7]
y_train = y[:7]
X_test = X[7:]
y_test = y[7:]

# Initialize the KNN Regressor with k=3
knn = KNNRegressor(k=3)

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Print predictions and actual values
print("Predictions:", y_pred)
print("Actual values:", y_test)

# Calculate Mean Squared Error (MSE) to evaluate the performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results for visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.plot(X, y, color='green', linestyle='dashed', label="True Line (For Visualization)")
plt.legend()
plt.title("KNN Regression (k=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


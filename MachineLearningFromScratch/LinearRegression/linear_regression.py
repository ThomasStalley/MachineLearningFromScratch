import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LinearRegression:
    """
    Linear Regression:
    -> Assuming data set has linear pattern (y = wx + b).
    -> We want to find the slope (and intercept) of the best linear fit.
    -> We do this by finding the minimum mean square error (MSE).
    -> MSE = "the average squared difference between the estimated values and the actual values".
    -> Minimum MSE means we have the best possible linear fit.

    Process:
    1. Predict results using y = wx + b (w_i = b_i = 0).
    2. Calculate error in prediction.
    3. Using gradient descent to find new weight and new bias values.
    4. Repeat n times.
    """

    def __init__(self, learning_rate=0.01, number_of_iterations=100):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """Fit the linear regression model to the training data."""
        number_of_samples, number_of_features = x.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0

        for _ in range(self.number_of_iterations):
            # Linear model:
            y_predictions = np.dot(x, self.weights) + self.bias

            # Compute gradients of the loss function, wrt weight and bias:
            dw = (2 / number_of_samples) * np.dot(x.T, (y_predictions - y))
            db = (2 / number_of_samples) * np.sum(y_predictions - y)

            # Gradient descent functions:
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, x):
        """Predict the target values using the linear regression model."""
        y_predictions = np.dot(x, self.weights) + self.bias
        return y_predictions


# ----------------------------------------------------------------------------------------------------------- Test Model
x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)
all_predictions = linear_regression_model.predict(x)
predictions = linear_regression_model.predict(x_test)
mse_all_data = np.mean((y - all_predictions) ** 2)
mse_test_data = np.mean((y_test - predictions) ** 2)

finer_linear_regression_model = LinearRegression(learning_rate=0.01, number_of_iterations=500)
finer_linear_regression_model.fit(x_train, y_train)
finer_predictions = finer_linear_regression_model.predict(x_test)
finer_mse_test_data = np.mean((y_test - finer_predictions) ** 2)

print(f"LinearRegression(learning_rate=0.01, number_of_iterations=100) -> MSE = {mse_test_data:.2f}")
print(f"LinearRegression(learning_rate=0.001, number_of_iterations=500) -> MSE = {finer_mse_test_data:.2f}")

# ------------------------------------------------------------------------------------------------- Plot Linear Fittings
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
font = {'family': 'serif'}

# All data:
axs[0].scatter(x, y, color=plt.get_cmap("viridis")(0.8), s=20, label="Data Points")
linear_fit = linear_regression_model.predict(x)
sorted_indices = np.argsort(x[:, 0])
axs[0].plot(x[sorted_indices, 0], linear_fit[sorted_indices], color="black", label="Linear Fitting")
axs[0].set_title(f"Linear Regression Fit - All Data - MSE = {mse_all_data:.2f}", fontdict=font)
axs[0].set_xlabel("Feature Value", fontdict=font)
axs[0].set_ylabel("Target Value", fontdict=font)
axs[0].legend()

# Test data:
axs[1].scatter(x_test, y_test, color=plt.get_cmap("viridis")(0.5), s=20, label="Test Data")
linear_fit_test = linear_regression_model.predict(x_test)
sorted_indices_test = np.argsort(x_test[:, 0])
axs[1].plot(x_test[sorted_indices_test, 0], linear_fit_test[sorted_indices_test], color="black", label="Linear Fitting")
axs[1].set_title(f"Linear Regression Fit - Test Data - MSE = {mse_test_data:.2f}", fontdict=font)
axs[1].set_xlabel("Feature Value", fontdict=font)
axs[1].set_ylabel("Target Value", fontdict=font)
axs[1].legend()

plt.tight_layout()
plt.show()

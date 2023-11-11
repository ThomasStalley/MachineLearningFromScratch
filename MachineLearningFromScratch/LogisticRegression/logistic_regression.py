import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LogisticRegression:
    """
    Logistic Regression:
    -> Create probabilities, instead of predicted outcome.
    -> The sigmoid function maps any real-valued number into the range (0, 1), which can correspond to a proability.
    -> So we can plug in a value/prediction to the sigmoid function, and get a corresponding probability.
    -> Optimisation algorithms, such as gradient descent, are used to find optimised model parameters.
    -> The optimisation algorithms iteratively adjust the parameters to minimize a cost function.
    -> The optimisation process enhances the model's ability to make accurate predictions.

    Training:
    1. Predict results using y = (1+exp(-wx+b))^(-1), with w_0 = b_0 = 0.
    2. Calculate error in prediction.
    3. Using gradient descent to find new weight and new bias values.
    4. Repeat n times.

    Testing:
    1. Plug in value from data point into sigmoid function.
    2. Choose label based on given probability.
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """Fit the logistic regression model to the training data."""
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Find linear predictions, then plug into sigmoid to get probabilities:
            linear_predictions = np.dot(x, self.weights) + self.bias
            sigmoid_probabilities = self._sigmoid_function(linear_predictions)

            # Compute gradients of the loss function, wrt weight and bias:
            dw = (1 / n_samples) * np.dot(x.T, (sigmoid_probabilities - y))
            db = (1 / n_samples) * np.sum(sigmoid_probabilities - y)

            # Gradient descent functions:
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, x):
        """Predict the target values using the logistic regression model."""
        y_linear_predictions = np.dot(x, self.weights) + self.bias
        y_sigmoid_probabilities = self._sigmoid_function(y_linear_predictions)
        predictions = [0 if i <= 0.5 else 1 for i in y_sigmoid_probabilities]
        return predictions

    def _sigmoid_function(self, x):
        """Create and return a sigmoid function."""
        sigmoid_function = (1 + np.exp(-x)) ** (-1)
        return sigmoid_function


# ----------------------------------------------------------------------------------------------------------- Test Model
breast_cancer_data = datasets.load_breast_cancer()
x, y = breast_cancer_data["data"], breast_cancer_data["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)
logistic_regression_model_predictions = logistic_regression_model.predict(x_test)
lrm_accuracy = (np.sum(logistic_regression_model_predictions == y_test) / len(y_test))
print(f"LogisticRegression(learning_rate=0.001, n_iterations=1000) -> accuracy =  {lrm_accuracy * 100:.2f}%")

logistic_regression_model_v2 = LogisticRegression(learning_rate=0.0001, n_iterations=5000)
logistic_regression_model_v2.fit(x_train, y_train)
logistic_regression_model_v2_predictions = y_pred = logistic_regression_model_v2.predict(x_test)
lrm_v2_accuracy = (np.sum(logistic_regression_model_v2_predictions == y_test) / len(y_test))
print(f"LogisticRegression(learning_rate=0.0001, n_iterations=5000) -> accuracy = {lrm_v2_accuracy * 100:.2f}%")

# ----------------------------------------------------------------------------------------------------- Plot Predictions
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Plot the predictions with the color coding:
colors = ["green" if prediction == truth else "red" for prediction, truth in zip(y_pred, y_test)]
plt.scatter(range(len(y_test)), logistic_regression_model_predictions, alpha=0.5, c=colors, edgecolors="w", s=100)
plt.yticks([0, 1], ['Malignant (Cancerous)', 'Benign (Non-Cancerous)'])

# Titles, labels and legend:
plt.title("Logistic Regression: Predicting if a tumor is cancerous.", fontsize=14)
plt.xlabel("Data Index", fontsize=12)
plt.ylabel("Tumor Classification", fontsize=12)
plt.scatter([], [], c="green", alpha=0.5, s=100, edgecolors="w", label="Correct Prediction")
plt.scatter([], [], c="red", alpha=0.5, s=100, edgecolors="w", label="Incorrect Prediction")
plt.legend(fontsize=10)
plt.show()

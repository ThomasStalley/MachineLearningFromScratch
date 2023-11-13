from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from MachineLearningFromScratch.DecisionTrees.decision_trees import DecisionTree


class RandomForest:
    """
    Random Forest:
    -> A collection of many decision trees, trained on different subsets of the same data.

    Training:
    -> Split the data into many random subsets, and create a decision tree for each.
    -> For each data point we now have a prediction from each tree:

    Testing:
    -> For classification we get the majority vote label.
    -> For regression we get the mean of all the predictions.
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, x, y):
        """Create a tree, given a random subset of training data, n times."""
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            x_sample, y_sample = self._bootstrap_samples(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        """"""
        predictions_by_each_tree = np.array([tree.predict(x) for tree in self.trees])
        predictions_for_each_datapoint = np.swapaxes(a=predictions_by_each_tree, axis1=0, axis2=1)
        prediction_for_each_datapoint = np.array([self._most_common_label(pr) for pr in predictions_for_each_datapoint])
        return prediction_for_each_datapoint

    def _bootstrap_samples(self, x, y):
        """"""
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return x[indices], y[indices]

    def _most_common_label(self, y):
        """Find the most common prediction in group of predictions."""
        label_counter = Counter(y)
        value = label_counter.most_common(n=1)[0][0]
        return value


# ----------------------------------------------------------------------------------------------------------- Test Model
breast_cancer_data = datasets.load_breast_cancer()
x, y = breast_cancer_data["data"], breast_cancer_data["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

random_forest_model = RandomForest()
random_forest_model.fit(x_train, y_train)
random_forest_model_predictions = y_pred = random_forest_model.predict(x_test)
random_forest_model_accuracy = (np.sum(y_test == random_forest_model_predictions) / len(y_test)) * 100
print(f"RandomForest(n_trees=10, max_depth=10, min_samples_split=2) -> accuracy = {random_forest_model_accuracy:.2f}%")

# ----------------------------------------------------------------------------------------------------- Plot Predictions
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Plot the predictions with the color coding:
colors = ["pink" if prediction == truth else "purple" for prediction, truth in zip(y_pred, y_test)]
plt.scatter(range(len(y_test)), y_pred, alpha=0.5, c=colors, edgecolors="w", s=100)
plt.yticks([0, 1], ["Malignant (Cancerous)", "Benign (Non-Cancerous)"])

# Titles, labels and legend:
plt.title("Random Forest: Predicting if a tumor is cancerous.", fontsize=14)
plt.xlabel("Data Index", fontsize=12)
plt.ylabel("Tumor Classification", fontsize=12)
plt.scatter([], [], c="pink", alpha=0.5, s=100, edgecolors="w", label="Correct Prediction")
plt.scatter([], [], c="purple", alpha=0.5, s=100, edgecolors="w", label="Incorrect Prediction")
plt.legend(fontsize=10)
plt.show()

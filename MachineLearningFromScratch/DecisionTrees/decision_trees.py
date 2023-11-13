from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
Tree Features:
-> root node = first node, containing all data points.
-> node = point containing data points, after one or more splittings of the data, due to decision.
-> terminal node = leaf node = the endpoint node, reached via a final decision (non leaf node is internal node).
-> branch = each node splits into two branches, after a decision.

Decisions:
-> Need to decide the order of features on which we split data on.
-> Need to decide the point within numerical features on which we split data on.
-> Need to decide when we stop splitting data.

Investigation:
-> Calculate information gain with each splitting.
-> Divide set with feature/value that gives most information gain.
-> Repeat this for all created branches, until a stopping criteria is reached.

Definitions:
-> Information Gain = IG = Entropy(parent) - weighted_average * Entropy(children)
-> Stopping Criteria = max depth, min number of samples, min impurity decrease (required level of entropy change).

Testing:
-> Follow tree until you reach a leaf node.
-> Return most common class label in this leaf node.
"""


# ----------------------------------------------------------------------------------------------------------- Node Class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Terminal node will have been given a value attribute, that is not None, which we check for here."""
        is_leaf_node = self.value is not None
        return is_leaf_node


# -------------------------------------------------------------------------------------------------- Decision Tree Class
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        """Grow a decision tree using given training data."""
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        """Recursively split the tree until stopping criteria is met, and we have a label to use as prediction."""
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria:
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Check stopping criteria not met, so we find the best splitting:
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature_index, best_threshold = self._best_split(x, y, feature_indices)

        # Aplpy best splitting to create child nodes:
        left_indices, right_indices = self._split(x_column=x[:, best_feature_index], split_threshold=best_threshold)
        left = self._grow_tree(x=x[left_indices, :], y=y[left_indices], depth=depth + 1)
        right = self._grow_tree(x=x[right_indices, :], y=y[right_indices], depth=depth + 1)
        return Node(feature=best_feature_index, threshold=best_threshold, left=left, right=right)

    def _most_common_label(self, y):
        """Find the most common label (in terminal node), to be used as prediction."""
        label_counter = Counter(y)
        value = label_counter.most_common(n=1)[0][0]
        return value

    def _best_split(self, x, y, feature_indices):
        """Iterate through features, and potential thresholds, to find ideal data splitting criteria."""
        best_information_gain = -1
        split_index, split_threshold = None, None

        # Iterate through features, where one will be chosen as best feature to split data upon:
        for feature_index in feature_indices:
            x_column = x[:, feature_index]

            # Iterate through potential thresholds, on which to split the data:
            thresholds = np.unique(x_column)
            for threshold in thresholds:

                # Calculate information gain, for each feature & threshold combination:
                information_gain = self._information_gain(y, x_column, threshold)
                if information_gain > best_information_gain:
                    # Best information gain so far, so we replace current best feature and threshold:
                    best_information_gain = information_gain
                    split_index = feature_index
                    split_threshold = threshold
        return split_index, split_threshold

    def _information_gain(self, y_parent, x_column, threshold):
        """Calculate information gain for given feature and threshold."""
        parent_entropy = self._entropy(y_parent)

        # Create children:
        left_indices, right_indices = self._split(x_column, threshold)
        if (len(left_indices) == 0) or (len(right_indices) == 0):
            information_gain = 0
            return information_gain

        # Calculate weighted average entropy of children:
        n_samples = len(y_parent)
        n_samples_l, n_samples_r = len(left_indices), len(right_indices)
        y_left, y_right = y_parent[left_indices], y_parent[right_indices]
        entropy_left, entropy_right = self._entropy(y_left), self._entropy(y_right)
        weighted_children_entropy = (n_samples_l / n_samples) * entropy_left + (n_samples_r / n_samples) * entropy_right

        # Calculate information gain:
        information_gain = parent_entropy - weighted_children_entropy
        return information_gain

    def _entropy(self, y):
        """Calculate the entropy of given dataset."""
        histogram = np.bincount(y)
        probability_distribution = histogram / len(y)
        entropy = -1 * np.sum([prob * np.log2(prob) for prob in probability_distribution if prob > 0])
        return entropy

    def _split(self, x_column, split_threshold):
        """Split the indices of a dataset into two groups, according to chosen threshold."""
        left_indices = np.argwhere(x_column <= split_threshold).flatten()
        right_indices = np.argwhere(x_column > split_threshold).flatten()
        return left_indices, right_indices

    def predict(self, x):
        """For each data point, travel the decision treem until terminal node reached."""
        tree_predictions = np.array([self._traverse_tree(x_value, self.root) for x_value in x])
        return tree_predictions

    def _traverse_tree(self, x_value, node):
        """If terminal node is reached return the label prediction, if not, travel further down the tree."""

        # Check if tree is ready to return prediction:
        if node.is_leaf_node():
            return node.value

        # See if datapoint follows left or right branch after splitting:
        if x_value[node.feature] <= node.threshold:
            return self._traverse_tree(x_value, node.left)
        return self._traverse_tree(x_value, node.right)

    def plot_tree(self, feature_names, node=None, depth=0, x_pos=0.5, x_scale=0.5, y_scale=0.5, y_node=0.05):
        """Plot the decision tree starting from a given node."""
        if node is None:
            node = self.root

        y_pos = 1 - depth * y_scale

        # If the node is a leaf, display its value:
        if node.is_leaf_node():
            leaf_label = {0: "Malignant", 1: "Benign"}.get(node.value)
            plt.text(
                x_pos,
                y_pos - y_node,
                leaf_label,
                horizontalalignment="center",
                verticalalignment="center",
                color="#006400",
                fontsize=8,
            )
            return

        # If the node is not a leaf, display its feature and threshold:
        feature_name = feature_names[node.feature]
        feature_label = f"Feature: {feature_name}\nThreshold: {node.threshold}"
        plt.text(x_pos, y_pos, feature_label, horizontalalignment="center", verticalalignment="center")

        # Recursively plot left and right branches with updated x positions:
        left_x_pos = x_pos - x_scale
        right_x_pos = x_pos + x_scale

        # Draw lines to children:
        plt.plot([x_pos, left_x_pos], [y_pos - y_scale / 10, y_pos - y_scale], color="brown", linestyle="-", alpha=0.7)
        plt.plot([x_pos, right_x_pos], [y_pos - y_scale / 10, y_pos - y_scale], color="brown", linestyle="-", alpha=0.7)

        # Recursively call plot_tree function for each branch:
        self.plot_tree(feature_names, node.left, depth + 1, left_x_pos, x_scale / 2, y_scale)
        self.plot_tree(feature_names, node.right, depth + 1, right_x_pos, x_scale / 2, y_scale)


# ----------------------------------------------------------------------------------------------------------- Test Model
breast_cancer_data = datasets.load_breast_cancer()
x, y = breast_cancer_data["data"], breast_cancer_data["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

decision_tree_model = DecisionTree(max_depth=10)
decision_tree_model.fit(x_train, y_train)
decision_tree_model_predictions = y_pred = decision_tree_model.predict(x_test)
decision_tree_model_accuracy = np.sum(y_test == decision_tree_model_predictions) / len(y_test)
print(f"DecisionTree(max_depth=10) accuracy: {decision_tree_model_accuracy * 100:.2f}%")

another_decision_tree_model = DecisionTree(max_depth=30)
another_decision_tree_model.fit(x_train, y_train)
another_decision_tree_model_predictions = more_y_pred = another_decision_tree_model.predict(x_test)
another_decision_tree_model_accuracy = np.sum(y_test == another_decision_tree_model_predictions) / len(y_test)
print(f"DecisionTree(max_depth=30) accuracy: {another_decision_tree_model_accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------------------------- Visualise Decision Tree
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
decision_tree_model.plot_tree(feature_names=breast_cancer_data.feature_names)
plt.axis('off')
plt.show()

# ----------------------------------------------------------------------------------------------------- Plot Predictions
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Plot the predictions with the color coding:
colors = ["orange" if prediction == truth else "blue" for prediction, truth in zip(y_pred, y_test)]
plt.scatter(range(len(y_test)), y_pred, alpha=0.5, c=colors, edgecolors="w", s=100)
plt.yticks([0, 1], ["Malignant (Cancerous)", "Benign (Non-Cancerous)"])

# Titles, labels and legend:
plt.title("Decision Tree: Predicting if a tumor is cancerous.", fontsize=14)
plt.xlabel("Data Index", fontsize=12)
plt.ylabel("Tumor Classification", fontsize=12)
plt.scatter([], [], c="orange", alpha=0.5, s=100, edgecolors="w", label="Correct Prediction")
plt.scatter([], [], c="blue", alpha=0.5, s=100, edgecolors="w", label="Incorrect Prediction")
plt.legend(fontsize=10)
plt.show()

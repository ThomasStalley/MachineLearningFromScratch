from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------------------------------- Create KNN Model
class KNN:
    """
    K Nearest Neighbours:
    -> Given a data point, calculate its distance from all other data points.
    -> Get the closest K points.
    -> Regression: Get the average of these K points.
    -> Classification: Get the label with the majority vote.
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        """Take in training data, and set them as class variables."""
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """Return prediction for each data point in given dataset, using knn method."""
        predictions = [self._predict(point) for point in x]
        return predictions

    def _predict(self, chosen_point):
        """Compute distances to all other points, get closest K points, and determine label with majority vote."""
        distances = [self._euclidean_distance(chosen_point, value) for value in self.x_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        majority_labels = Counter(k_nearest_labels).most_common()
        majority_label = majority_labels[0][0]
        return majority_label

    def _euclidean_distance(self, x1, x2):
        """Calculate euclidean distance between two data points."""
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance


# ---------------------------------------------------------------------------------------------------- Getting Iris Data
iris_dataset = datasets.load_iris()
x, y = iris_dataset["data"], iris_dataset["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# ----------------------------------------------------------------------------- Use KNN Model for Species Classification
classifier = KNN(k=5)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"KNN classifier model accuracy: {100 * accuracy:.2f}%")

# ----------------------------------------------------------------------------------------- Plotting Prediction vs Truth
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

rows = cols = [0, 1]
titles = ['Predicted Species', 'True Species']
x_labels = y_labels = ['Petal length (cm)', 'Petal width (cm)', 'Sepal length (cm)', 'Sepal width (cm)']
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

for row_index in rows:
    for col_index in cols:
        ax = axes[row_index, col_index]
        x_index, y_index = (2, 3) if row_index == 0 else (0, 1)
        c = predictions if col_index == 0 else y_test
        title = titles[col_index] if row_index == 0 else titles[1 - col_index]

        # Scatters:
        ax.set_title(title, fontstyle='italic', fontweight='bold', fontfamily='serif')
        scatter = ax.scatter(x_test[:, x_index], x_test[:, y_index], c=c, cmap=cmap, edgecolor='k', s=50)
        ax.set_xlabel(iris_dataset['feature_names'][x_index], fontfamily='serif')
        ax.set_ylabel(iris_dataset['feature_names'][y_index], fontfamily='serif')

        # Legends:
        handles, labels = scatter.legend_elements()
        legend_labels = iris_dataset['target_names']
        legend_properties = {'family': 'serif'}
        ax.legend(handles, legend_labels, title="", prop=legend_properties)

plt.tight_layout()
plt.show()

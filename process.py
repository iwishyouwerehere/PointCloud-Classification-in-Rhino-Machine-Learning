import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight
from scipy.spatial.distance import pdist
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
# Feature engineering functions
from sklearn import tree

def centroid(points):
    return np.mean(points, axis=0)

def min_max_range(points):
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    range_values = max_values - min_values
    return min_values, max_values, range_values

def bounding_box_volume(points):
    _, _, range_values = min_max_range(points)
    return np.prod(range_values)

def average_point_distance(points):
    distances = pdist(points)
    return np.mean(distances)

def eigenvalues(points):
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    return sorted(eigenvalues, reverse=True)

def thickness_range(points):
    min_values, max_values, _ = min_max_range(points)
    thickness_range_ = max_values[2] - min_values[2]
    return thickness_range_

def extract_features(sample):
    points = np.array(sample['points'])
    centroid_ = centroid(points)
    min_values, max_values, range_values = min_max_range(points)
    bbox_volume = bounding_box_volume(points)
    avg_point_dist = average_point_distance(points)
    eigenvalues_ = eigenvalues(points)
    thickness_range_ = thickness_range(points)

    features = np.hstack([centroid_, min_values, max_values, range_values, bbox_volume, avg_point_dist, eigenvalues_, thickness_range_])
    return features



# Load dataset

with open('dataset.json', 'r') as f:
    data = json.load(f)

# Preprocess dataset
X = np.array([extract_features(sample) for sample in data])
y = np.array([sample['label'] for sample in data])

# Analyze class distribution
print("Class distribution:\n", pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = class_weight.compute_sample_weight('balanced', y_train)

clf = RandomForestClassifier(n_estimators=200, random_state=42*4, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# Visualize a single decision tree from the Random Forest
# Visualize a single decision tree from the Random Forest
single_tree = clf.estimators_[0]  # You can change the index to see different trees

plt.figure(figsize=(30, 15))
tree.plot_tree(single_tree,
               feature_names=['centroid_x', 'centroid_y', 'centroid_z',
                              'min_x', 'min_y', 'min_z',
                              'max_x', 'max_y', 'max_z',
                              'range_x', 'range_y', 'range_z',
                              'bbox_volume', 'avg_point_dist',
                              'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
                              'thickness_range'],
               class_names=list(np.unique(y).astype(str)),
               filled=True,
               impurity=False,
               proportion=True,
               fontsize=9)
plt.show()

print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

# Feature Importance
print("Feature Importance:", clf.feature_importances_)

# Cross-Validation
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Cross-Validation Mean Score:", np.mean(scores))

# Use the model for inference on a new sample
joblib.dump(clf, 'type.joblib')

with open('target.json', 'r') as f:
    custom_data = json.load(f)

custom_points = custom_data[0]['points']
custom_X = extract_features({'points': custom_points})
print('Custom point cloud shape:', custom_X.shape)
custom_X = custom_X.reshape(1, -1)  # Reshape it to have the same format as the input data during training

custom_pred = clf.predict(custom_X)
print('Predicted category:', custom_pred[0])

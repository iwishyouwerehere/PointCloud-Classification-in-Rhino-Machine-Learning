from flask import Flask, request, jsonify
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
from sklearn import tree
import logging
from datetime import datetime

# Feature engineering functions
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

    features = np.hstack([
        centroid_,
        min_values,
        max_values,
        range_values,
        bbox_volume,
        avg_point_dist,
        eigenvalues_,
        thickness_range_
    ])
    return features

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)

# Load the trained model
clf = joblib.load('type.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get client's IP address
    client_ip = request.remote_addr

    # Get current date and time
    current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    # Load the new sample from the POST request
    custom_data = request.json
    custom_points = custom_data[0]['points']
    custom_X = extract_features({'points': custom_points})
    custom_X = custom_X.reshape(1, -1)

    # Use the model to predict the label for the new sample
    custom_pred = clf.predict(custom_X)
    prediction = str(custom_pred[0])

    # Log and print the information
    log_message = (
        f"IP del client: {client_ip} | "
        f"Data e ora: {current_time} | "
        f"Richiesta: {custom_data} | "
        f"Risposta: {prediction}"
    )
    app.logger.info(log_message)
    print(log_message)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6666, debug=True)

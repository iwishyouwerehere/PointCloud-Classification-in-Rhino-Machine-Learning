PointCloud-Classification
This repository contains a machine learning project that uses a Random Forest Classifier to categorize point clouds. The project includes feature extraction methods, model training, visualization, performance evaluation, and inference on new data.

Project Overview
The project involves the following steps:

Feature Extraction: We extract various features from the point cloud data such as centroid, min-max range, bounding box volume, average point distance, eigenvalues, and thickness range.

Data Preprocessing: The extracted features are then preprocessed and split into training and testing sets.

Model Training: A Random Forest Classifier is trained on the preprocessed data. The model is trained with balanced class weights to handle imbalanced datasets.

Model Evaluation: The trained model is evaluated on the test set. The evaluation metrics include a classification report and accuracy score.

Visualization: A single decision tree from the Random Forest is visualized for better understanding of the model.

Feature Importance: The importance of each feature used by the model is calculated and displayed.

Cross-Validation: The model's performance is further evaluated using cross-validation.

Inference: The trained model is used to predict the category of a new point cloud sample.

Usage
To use this project, you need to have Python installed along with the following libraries:

numpy
pandas
scikit-learn
matplotlib
joblib
scipy
You can install these libraries using pip:



pip install numpy pandas scikit-learn matplotlib joblib scipy
After installing the dependencies, you can run the script using Python:


python process2scikitattempt.py
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

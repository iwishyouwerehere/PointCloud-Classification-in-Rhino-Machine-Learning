PointCloud-Classification
This repository contains a machine learning project that uses a Random Forest Classifier to categorize point clouds. Point clouds are collections of points in 3D space that represent the shape of an object or a scene. They are often obtained from sensors such as LiDAR or depth cameras. Point cloud classification is the task of assigning a label to a point cloud based on its shape or content.



Project Overview
The project involves the following steps:

Feature Extraction: We extract various features from the point cloud data such as centroid, min-max range, bounding box volume, average point distance, eigenvalues, and thickness range. These features capture the geometric and statistical properties of the point clouds and help the model distinguish between different categories.
Data Preprocessing: The extracted features are then preprocessed and split into training and testing sets. We also normalize the features to have zero mean and unit variance.

Model Training: A Random Forest Classifier is trained on the preprocessed data. The model is trained with balanced class weights to handle imbalanced datasets. A Random Forest Classifier is an ensemble of decision trees that can handle high-dimensional and non-linear data.

Model Evaluation: The trained model is evaluated on the test set. The evaluation metrics include a classification report and accuracy score. The classification report shows the precision, recall, f1-score, and support for each class. The accuracy score shows the overall percentage of correct predictions.

Visualization: A single decision tree from the Random Forest is visualized for better understanding of the model. The visualization shows the splitting criteria and the class distribution at each node of the tree.

Feature Importance: The importance of each feature used by the model is calculated and displayed. The importance indicates how much each feature contributes to the model’s performance. The higher the importance, the more relevant the feature is for the classification task.

Cross-Validation: The model’s performance is further evaluated using cross-validation. Cross-validation is a technique that splits the data into multiple folds and trains and tests the model on each fold. This helps to reduce overfitting and estimate the generalization error of the model.

Inference: The trained model is used to predict the category of a new point cloud sample. The prediction is based on the features extracted from the sample.


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
This project is licensed under the MIT License. 

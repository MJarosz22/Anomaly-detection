# Anomaly Detection
This project is designed to detect anomalies in a given dataset using the Isolation Forest algorithm. The data is first split into training and testing sets, and the model is trained on the training data. The trained model is then used to make predictions on the test data, with predictions of 1 indicating an anomaly and 0 indicating normal behavior. The model's performance is also evaluated on a separate validation set using the F1 score.

### Prerequisites
This project requires Python 3 and the following libraries:

NumPy
pandas
scikit-learn

### Usage
To use the anomaly detection model, run the anomaly_detection.py script. This will train the model on the training data, make predictions on the test data, and evaluate the model's performance on the validation data. The predictions and evaluation results will be wriiten to a csv file.

### Data
The data used in this project is provided in three CSV files:

train_data.csv: contains the training data
test_data.csv: contains the test data for which predictions will be made
val_data.csv: contains the validation data for evaluating the model's performance
The data consists of multiple rows, each with multiple columns representing different features. The last column of the train_data.csv and val_data.csv files contains a label indicating whether the row represents normal behavior (0) or an anomaly (1).

### Methodology
The Isolation Forest algorithm is used to detect anomalies in the data. It works by training a number of decision trees on the data and using the trees to isolate individual data points by randomly selecting a feature and split value. Points that are isolated faster are more likely to be anomalies.

### Authors
Marcin Jarosz - Initial work - MJarosz22

import csv
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score


def extract_time(time):
    format = '%d/%m/%Y %I:%M:%S %p'
    if(time == "Timestamp"):
        return 0
    time = time.strip()
    date = datetime.strptime(time, format).timestamp()
    return date


def load_data(file):
    rows = []
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        count = 0
        for row in csv_reader:
            data_point = []
            for i in row:
                data_point.append(i)
            rows.append(data_point)

    for entry in rows:
        entry[1] = extract_time(entry[1])
    temp = np.array(rows[1:])
    return temp


def create_submission(submission, file):
    with open(file, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(submission)


train_data = np.array(load_data("data/train_data.csv")[:, 1:-1], dtype=float)
ids = load_data("data/test_data.csv")[:, 0]
test_data = np.array(load_data("data/test_data.csv")[:, 1:], dtype=float)
val_data = np.array(load_data("data/val_data.csv")[:, 1:-1], dtype=float)

model = IsolationForest(n_estimators=26, max_samples='auto', contamination=float(0.006))
model.fit(train_data)


anomaly = model.predict(test_data)
anomaly[anomaly == 1] = 0
anomaly[anomaly == -1] = 1

predicted = model.predict(val_data)
predicted[predicted == 1.] = 0.
predicted[predicted == -1.] = 1.

actual = np.array(load_data("data/val_data.csv")[:, -1], dtype=float)
f1 = f1_score(actual, predicted, average='micro')
print("F1 score: ", f1)

res = np.vstack((ids, anomaly)).T
create_submission(res, "submission.csv")
# the submission has to be reformatted manually!
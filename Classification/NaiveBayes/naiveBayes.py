'''
Naive Bayes (Gaussian) Classification
Demonstration of a Gaussian Naive Bayes classification process.

TODO: Actually run this to see if this works, I wrote this on my non-linux machine
so I have no idea whether if its going to work... (´・ω・`)
'''
import os
import numpy as np
import collections
from sklearn.naive_bayes import GaussianNB

# Kidney Disease Sample Dataset
file_dir = '../../datasets/kidney_disease/Kidney-Disease.csv'
data = np.genfromtxt(file_dir, delimiter=',', dtype="|S10")

# Split data into features and labels
data = np.delete(data, (0), axis=0)   # Remove first column label row
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Initialize GaussianNB
gbn = GaussianNB()

# Generate the Naive Bayes Model
gbn_model = gnb.fit(X_train, y_train)

# Generate Predictions and Compute Accuracy
y_pred = gbn_model.predict(y_test)
acc = (y_test == y_pred).sum() / float(y_test)

# Display Error Rate
print 'Mislabeled Data Points: ' + str((y_test != y_pred).sum()) + '/' + str(y_test)
print 'Accuracy: ' + acc

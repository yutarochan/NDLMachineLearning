'''
ZeroR Classification
Demonstration of a ZeroR classification process.
'''
import os
import numpy as np
import collections
from StringIO import StringIO

# Import Dataset into Numpy Array
# <Sample Dataset 1>
# file_dir = '../../datasets/breast-cancer-wisconson/wdbc.data'
# with open(file_dir, 'r') as data: raw_data = data.read()
# data = np.genfromtxt(StringIO(raw_data), delimiter=',', dtype="|S10")

file_dir = '../../datasets/kidney_disease/Kidney-Disease.csv'
data = np.genfromtxt(file_dir, delimiter=',', dtype="|S10")

np.delete(data, 1, 0)   # Remove column labels
labels = data[:,24]

'''
# Split data into train and test sets
np.random.shuffle(labels)                   # Randomly shuffle the data
train, test = np.array_split(labels, 2)     # Split to train and test

# Frequency Computations
train_freq = collections.Counter(train)     # Generate frequency counts
result = train_freq.most_common(1)[0][0]    # Get the most common frequency
'''
print 'Training Set Frequency: ' + str(labels)
print 'Result (Mode): ' + str(result)
print

# Test against test set
test_freq = collections.Counter(test)
acc = test_freq[result] / float(len(test_freq))

print 'Test Set Frequency: ' + str(test_freq)
print 'Accuracy: ' + str(acc) + "%"

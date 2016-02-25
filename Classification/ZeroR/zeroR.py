'''
ZeroR Classification
Demonstration of a ZeroR classification process.
'''
import numpy as np
import collections

# Kidney Disease Sample Dataset
file_dir = '../../datasets/kidney_disease/Kidney-Disease.csv'
data = np.genfromtxt(file_dir, delimiter=',', dtype="|S10")

data = np.delete(data, (0), axis=0)   # Remove column
labels = data[:,24]

# Split data into train and test sets
np.random.shuffle(labels)                   # Randomly shuffle the data
train, test = np.array_split(labels, 2)     # Split to train and test

# Frequency Computations
train_freq = collections.Counter(train)     # Generate frequency counts
result = train_freq.most_common(1)[0][0]    # Get the most common frequency

# Display Training Model
print '[ZeroR Generated Model]'
print 'Training Set Frequency: ' + str(train_freq)
print 'Result (Mode): ' + str(result)
print

# Test against test set
test_freq = collections.Counter(test)
acc = test_freq[result] / float(len(test))

# Display Test Results
print '[ZeroR Tested Models]'
print 'Test Set Frequency: ' + str(test_freq)
print 'Accuracy: ' + str(acc)  # Should be ~28% if done correctly

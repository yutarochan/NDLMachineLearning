import numpy as np
import collections

file_dir = 'Kidney-Disease.csv'
data = np.genfromtxt(file_dir, delimiter=',', dtype='|S10')

data = np.delete(data, (0), axis=0)
labels = data[:,24]

# Split data into train and test sets
np.random.shuffle(labels)                   # Randomly shuffle the data
train, test = np.array_split(labels, 2)     # Split to train and test

# Frequency Computations
train_freq = collections.Counter(train)     # Generate frequency counts
result = train_freq.most_common(1)[0][0]

# Display Training Model
print '[ZeroR Generated Model]'
print 'Training Set Frequency: ' + str(train_freq)
print 'Result (Mode): ' + str(result)
print

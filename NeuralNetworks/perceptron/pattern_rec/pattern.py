from Perceptron import Perceptron
import numpy as np

# Initialize Input Parameters
input = np.array([50, -12, 1])

# Feedfoward Perceptron Test
print 'Initial Feedforward Testing'
p = Perceptron(len(input))              # Initialize Perceptron Instance
result = p.feedforward(input)           # Feed input into perceptron

# Display Results
print 'Input: ' + str(input)
print 'Answer: ' + str(result)
print

# Training our Perceptron
# Perceptron Parameters
input = np.array([ [50, -12, 1], [10, 10, 1], [-10, 30, 1] ])
label = np.array([ 1, -1, 1 ])

# Training
p2 = Perceptron(len(input))
result = p2.train(input, label)

# Testing
test = np.array([ [30, -10, 1], [9, 13, 1], [-12, 22, 1] ])

print 'Testing #1: ' + str([30, -10, 1])
print 'Result #1: ' + str(p2.feedforward([30, -10, 1]))

print 'Testing #2: ' + str([10, 15, 1])
print 'Result #2: ' + str(p2.feedforward([10, 15, 1]))

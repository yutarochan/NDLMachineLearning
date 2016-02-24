'''
Simple Perceptron
A demonstration of a single toy perceptron unit and how weights are computed.
Based on the example from 10.2 Perceptron from http://natureofcode.com/book/chapter-10-neural-networks/
'''
import numpy as np

# Activation Function
def activate(value):
    if value > 0: return 1
    else: return -1

# Setup Parameters for Perceptron
input = [12, 4]     # Input parameters for perceptron
weights = [.5, -1]  # Weight parameters for perceptron

# Take the "dot product" of input and weightcl
result = np.dot(input, weights)

# Display the result of perceptron
# The total should come out as 2.0
print result

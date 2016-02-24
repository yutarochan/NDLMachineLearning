'''
Perceptron Class
Demonstration of perceptron class implemented from:
http://natureofcode.com/book/chapter-10-neural-networks/
'''
import numpy as np
from numpy.random import uniform

class Perceptron:
    # Class Constructor
    def __init__(self, n, c=0.01):
        # Initialize weight matrix with random weights
        self.weights = np.random.uniform(-1, 1, n)
        self.rate = c;

    # Activation Function
    def activate(self, value):
        if value > 0: return 1
        else: return -1

    # Feedfoward Function for Perceptron
    # Takes the dot product between the input and the weight vectors
    def feedforward(self, inputs):
        return self.activate(np.dot(inputs, self.weights))

    # Train Perceptron
    def train(self, inputs, label):
        print 'Training...\n'
        print 'Weight Before: '
        print self.weights

        '''
        # Iterate over each training example in the input matrix
        for i in range(0, len(inputs)):
            guess = self.feedforward(inputs)    # Guess from our inputs
            error = label - guess               # Derive error of perceptron
            self.weights =                      # Adjust Perceptron Weights
        '''
        
        print 'Weight After: '
        print self.weights

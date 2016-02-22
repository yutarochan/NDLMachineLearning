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
        self.weights = np.random.uniform(-1, 1, n)   # Initialize weight matrix with ranom weights
        self.rate = c;

    # Activation Function
    def activate(self, value):
        if value > 0: return 1
        else: return -1

    # Feedfoward Function for Perceptron
    def feedforward(self, inputs):
        return self.activate(np.multiply(inputs, self.weights).sum())

    # Train Perceptron
    def train(self, inputs, label):
        guess = self.feedforward(inputs)                                                           # Guess from our inputs
        error = label - guess                                                                      # Derive error of perceptron
        self.weights =  np.add(np.multiply(self.rate, np.multiply(error, inputs)), self.weights)      # Adjust Perceptron Weights

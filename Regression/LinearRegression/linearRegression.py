import matplotlib.pyplot as plt
import numpy as np

# Warmup Exercise - Create a 5x5 Identity Matrix
A = np.identity(5)
print A

# Load Dataset
file_dir = '../../datasets/housing_price/housing1.txt'
data = np.genfromtxt(file_dir, delimiter=',')
print data

plt.plot(data, '.')
plt.show()

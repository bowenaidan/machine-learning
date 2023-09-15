#EECS 658 Assignment 1
#Checks version and prints hello world
#Outputs:
#Hello, World!
#Python, Scipy, Numpy, Pandas, and scikit-learn versions
#Author: Aidan Bowen
#Date: 8-31-2023

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

#hello world
print('Hello, World!\n')
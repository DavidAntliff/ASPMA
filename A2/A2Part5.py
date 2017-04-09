import numpy as np
from A2Part3 import DFT

"""
A2-Part-5: Compute the magnitude spectrum (Optional)

Write a function that computes the magnitude spectrum of an input sequence x of length N. The 
function should return an N point magnitude spectrum with frequency index ranging from 0 to N-1.

The input argument to the function is a numpy array x and the function should return a numpy array of the 
magnitude spectrum of x.

EXAMPLE: If you run your function using x = np.array([1, 2, 3, 4]), the function should return the following 
numpy array magX: [array([10.0, 2.82842712, 2.0, 2.82842712])
"""
def genMagSpec(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array
        magX (numpy array) = The magnitude spectrum of the input sequence x
                             (length N)
    """
    X = DFT(x)
    magX = np.abs(X)
    return magX

def test_genMagSpec():
    assert np.allclose(genMagSpec(np.array([1, 2, 3, 4])), np.array([10.0, 2.82842712, 2.0, 2.82842712]))

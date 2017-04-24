import numpy as np
import scipy
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

assert np.version.version == "1.11.0"
assert scipy.version.version == "0.17.0"

""" 
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window 
type and its length (M). The function should return the samples corresponding to the main lobe in 
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this 
part, N need not be a power of 2). 

The input arguments to the function are the window type (window) and the length of the window (M). 
The function should return a numpy array containing the samples corresponding to the main lobe of 
the window. 

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe. 

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the 
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas 
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices 
have to be obtained for each possible window types separately (as they differ across different 
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16 
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers. 
If you find it difficult to concatenate the two halves of the main lobe, you can first center the 
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy 
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array 
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array 
should contain 33 samples.

"""
def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude 
            spectrum of the window in decibels (dB).
    """

    logger.debug("window {}, M {}".format(window, M))
    w = get_window(window, M)         # get the window 
    N = 8 * M

    X = fft(w, N)

    Xshift = fftshift(X)
    plt.plot(abs(Xshift))
    plt.plot(20*np.log10(abs(Xshift)))

    m1 = find_minima(abs(Xshift), N/2, -1)
    m2 = find_minima(abs(Xshift), N/2, +1)
    logger.info("m1 {}, m2 {}".format(m1, m2))

    Xlobe = Xshift[m1:m2+1]
    absXlobe = abs(Xlobe) + eps
    #absXlobe[absXlobe < eps] = eps  # avoid log(0)
    logger.info("len absXlobe {}".format(len(absXlobe)))
    return 20 * np.log10(absXlobe)


def find_minima(x, start, direction=1):
    y = 0
    i = start + direction
    while i < len(x) and i >= 0 and x[i] < x[i - direction]:
        i += direction
    return i - direction


def get_test_case(part_id, case_id):
    import loadTestCases
    testcase = loadTestCases.load(part_id, case_id)
    return testcase


def test_case_1():
    testcase = get_test_case(1, 1)
    main_lobe = extractMainLobe(**testcase['input'])
    plt.figure()
    plt.plot(main_lobe)
    plt.plot(testcase['output'])
    plt.show()
    assert np.allclose(testcase['output'], main_lobe, atol=1e-6, rtol=0)


def test_case_2():
    testcase = get_test_case(1, 2)
    main_lobe = extractMainLobe(**testcase['input'])
    assert np.allclose(testcase['output'], main_lobe, atol=1e-6, rtol=0)


def test_case_3():
    testcase = get_test_case(1, 3)
    main_lobe = extractMainLobe(**testcase['input'])
    assert np.allclose(testcase['output'], main_lobe, atol=1e-6, rtol=0)

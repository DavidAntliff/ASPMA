# Copied from 4P1 @ 11:30.
# Demonstrates how a single windowed sinewave results in a spectrum where the
# window spectrum is convolved with the sinewave spectrum.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys
sys.path.append('../../software/models/')
import dftModel as DFT


def main():
    fs = 44100
    f = 5000.0
    M = 101
    x = np.cos(2*np.pi*f*np.arange(M)/float(fs))
    N = 512
    # w = get_window('hanning', M)
    w = get_window('hamming', M)
    mX, pX = DFT.dftAnal(x, w, N)

    # plt.plot(np.arange(0, fs/2, fs/float(N)), mX-max(mX))  # incorrect number of array elements
    plt.plot(np.linspace(0, fs/2, N/2+1), mX-max(mX))
    plt.show()
    return locals()

# To make main() locals available to ipython prompt:
#
# > import test
# > llc = test.main()
# > locals().update(llc)
# > plot(window)
#

if __name__ == "__main__":
    main()

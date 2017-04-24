# Copied from 4P1 @ 00:50.
# Shows tradeoff between windows and lobe width / sidelobe level
#

import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt


def main():
    M = 63
    # window = get_window('hanning', M)
    # window = get_window('hamming', M)
    # window = get_window('blackman', M)
    window = get_window('blackmanharris', M)
    hM1 = int(math.floor((M+1)/2))  # handle even and odd cases of M
    hM2 = int(math.floor(M/2))

    N = 512
    hN = N/2  # video annotation says "hN = N/2 - 1" - what does this mean?
    fftbuffer = np.zeros(N)
    fftbuffer[:hM1] = window[hM2:]
    fftbuffer[N-hM2:] = window[:hM2]

    X = fft(fftbuffer)
    absX = abs(X)
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # avoid log(0)
    mX = 20*np.log10(absX)
    pX = np.angle(X)

    mX1 = np.zeros(N)
    pX1 = np.zeros(N)
    mX1[:hN] = mX[hN:]
    mX1[N-hN:] = mX[:hN]
    pX1[:hN] = pX[hN:]
    pX1[N-hN:] = pX[:hN]

    plt.plot(np.arange(-hN, hN) / float(N) * M, mX1 - max(mX1))
    plt.axis([-20, 20, -80, 0])
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

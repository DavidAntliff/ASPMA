import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import stft as STFT


def main():
    inputFile = "../../sounds/flute-A4.wav"
    window = "hamming"
    M = 801
    N = 1024
    H = 400

    fs, x = UF.wavread(inputFile)

    w = get_window(window, M)

    mX, pX = STFT.stftAnal(x, w, N, H)

    plt.pcolormesh(np.transpose(mX))

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

import numpy as np
from scipy.signal import get_window, resample
from scipy.fftpack import fft
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import dftModel as DFT

fs, x = UF.wavread('../../sounds/oboe-A4.wav')

M = N = 512
w = get_window('hanning', M)
xw = x[10000:10000+M] * w

filter = get_window('hamming', 30) * -60.0

mX, pX = DFT.dftAnal(xw, w, N)

centerbin = 40
mY = np.copy(mX)
mY[centerbin-15:centerbin+15] = mX[centerbin-15:centerbin+15] + filter

y = DFT.dftSynth(mY, pX, N) * sum(w)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(xw)
plt.plot(y)

plt.figure()
plt.plot(filter)

plt.figure()
plt.plot(mX)
plt.plot(mY)

plt.show()

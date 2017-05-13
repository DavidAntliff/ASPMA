import numpy as np
from scipy.signal import get_window, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF

fs, x = UF.wavread('../../sounds/ocean.wav')
M = N = 256
stocf = 0.2
w = get_window('hamming', M)
xw = x[10000:10000+M] * w
X = fft(xw)
mX = 20 * np.log10(abs(X[:N/2]))
mXenv = resample(np.maximum(-200, mX), N/2*stocf)

mY = resample(mXenv, N/2)
pY = 2 * np.pi * np.random.rand(N/2)
Y = np.zeros(N, dtype=complex)
Y[:N/2] = 10**(mY / 20.0) * np.exp(1j * pY)
Y[N/2+1:] = 10**(mY[:0:-1] / 20.0) * np.exp(-1j * pY[:0:-1])
# Y[N/2:] = 10**(mY[::-1] / 20.0) * np.exp(-1j * pY[::-1])  # forum suggests this
y = np.real(ifft(Y))

import matplotlib.pyplot as plt
plt.plot(mX)
plt.plot(mY)
plt.show()

plt.plot(xw)
plt.plot(y)
plt.show()

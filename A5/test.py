import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models'))
import dftModel as DFT
import utilFunctions as UF

# For sine-440.wav:
# (fs, x) = UF.wavread('../../sounds/sine-440.wav')
# M = 501
# N = 512
# N = 2048  # better freq resolution for sine-440.wav

# For sine-440-490.wav
(fs, x) = UF.wavread('../../sounds/sine-440-490.wav')
M = 3528  # M = 4 bins * 44100 / (490-440)
N = 4096  # N > M

t = -20  # threshold
w = get_window('hamming', M)
x1 = x[0.8*fs:0.8*fs+M]
mX, pX = DFT.dftAnal(x1, w, N)
ploc = UF.peakDetection(mX, t)
pmag = mX[ploc]

freqaxis = fs * np.arange(N/2+1)/float(N)
plt.plot(freqaxis, mX)
plt.plot(fs * ploc / float(N), pmag, marker='x', linestyle='')

# quadratic interpolation:
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
plt.plot(fs * iploc / float(N), ipmag, marker='x', linestyle='')


plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models'))
import utilFunctions as UF
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft

fs = 44100
Ns = 512
hNs = Ns / 2
H = Ns / 4
ipfreq = np.array([4000.0])
ipmag = np.array([0.0])
ipphase = np.array([0.0])
Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)
y = np.real(ifft(Y))

# synthesis window = triangle window / blackman-harris window
sw = np.zeros(Ns)
ow = triang(Ns / 2)
sw[hNs-H:hNs+H] = ow
bh = blackmanharris(Ns)
bh = bh / sum(bh)
sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

yw = np.zeros(Ns)
yw[:hNs-1] = y[hNs+1:]
yw[hNs-1:] = y[:hNs+1]
yw *= sw

#plt.plot(abs(Y))
#plt.plot(y)  # synthesised result (note blackmanharris window shape)
#plt.plot(yw)  # scale to triangle window for 50% overlap
plt.show()

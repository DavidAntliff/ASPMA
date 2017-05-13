import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import stochasticModel as STM

fs, x = UF.wavread('../../sounds/ocean.wav')
H = 128
stocf = 0.2
stocEnv = STM.stochasticModelAnal(x, H, H * 2, stocf)

import matplotlib.pyplot as plt
import numpy as np
plt.pcolormesh(np.transpose(stocEnv))
plt.show()

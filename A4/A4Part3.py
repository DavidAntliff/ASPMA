import os
import sys
import numpy as np
import scipy
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

assert np.version.version == "1.11.0"
assert scipy.version.version == "0.17.0"

"""
A4-Part-3: Computing band-wise energy envelopes of a signal

Write a function that computes band-wise energy envelopes of a given audio signal by using the STFT.
Consider two frequency bands for this question, low and high. The low frequency band is the set of 
all the frequencies between 0 and 3000 Hz and the high frequency band is the set of all the 
frequencies between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 
At a given frame, the value of the energy envelope of a band can be computed as the sum of squared 
values of all the frequency coefficients in that band. Compute the energy envelopes in decibels. 

Refer to "A4-STFT.pdf" document for further details on computing bandwise energy.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N) and hop size (H). The function should return a numpy 
array with two columns, where the first column is the energy envelope of the low frequency band and 
the second column is that of the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases.You can clearly notice the sharp attacks and decay of the piano notes for test case 1 
(See figure in the accompanying pdf). You can compare this with the output from test case 2 that 
uses a larger window. You can infer the influence of window size on sharpness of the note attacks 
and discuss it on the forums.
"""
def computeEngEnv(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, 
                hamming, blackman, blackmanharris)
            M (integer): analysis window size (odd positive integer)
            N (integer): FFT size (power of 2, such that N > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a numpy array engEnv with shape Kx2, K = Number of frames
            containing energy envelop of the signal in decibles (dB) scale
            engEnv[:,0]: Energy envelope in band 0 < f < 3000 Hz (in dB)
            engEnv[:,1]: Energy envelope in band 3000 < f < 10000 Hz (in dB)
    """
    _, _, env = compute_eng_env(inputFile, window, M, N, H)
    return env


def compute_eng_env(inputFile, window, M, N, H):
    fs, x = UF.wavread(inputFile)
    w = get_window(window, M)

    mX, pX = stft.stftAnal(x, w, N, H)
    mXlinear = 10.0 ** (mX / 20.0)

    # Get an array of indices for bins within each band range:

    # Using list comprehension:
    # band_low_bins = np.array([ k for k in range(N) if 0 < k * fs / N < 3000.0])
    # band_high_bins = np.array([ k for k in range(N) if 3000.0 < k * fs / N < 10000.0])

    # Using np.where():
    bins = np.arange(0, N) * fs / N
    band_low_bins = np.where((bins > 0) & (bins < 3000.0))[0]
    band_high_bins = np.where((bins > 3000) & (bins < 10000.0))[0]

    num_frames = mX.shape[0]
    env = np.zeros(shape=(num_frames, 2))

    for frame in range(num_frames):
        env[frame, 0] = 10.0 * np.log10(sum(mXlinear[frame, band_low_bins] ** 2))
        env[frame, 1] = 10.0 * np.log10(sum(mXlinear[frame, band_high_bins] ** 2))

    plot_spectrogram_with_energy_envelope(mX, env, M, N, H, fs, 'mX ({}), M={}, N={}, H={}'.format(inputFile, M, N, H))

    return fs, mX, env


def plot_spectrogram_with_energy_envelope(mX, env, M, N, H, fs, title):
    assert mX.shape[0] == env.shape[0]
    num_frames = mX.shape[0]

    frmTime = H * np.arange(num_frames) / float(fs)
    binFreq = np.arange(N / 2 + 1) * float(fs) / N

    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.title("Spectrogram")
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX), cmap='jet')
    plt.autoscale(tight=True)
    plt.ylim([0, 10000])
    # plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(3, 1, 2)
    plt.title("Energy Envelopes")
    plt.plot(frmTime, env[:, 0], 'r', label='Low')
    plt.plot(frmTime, env[:, 1], 'b', label='High')
    plt.xlabel("Time (sec)")
    plt.ylabel("Energy (dB)")
    plt.legend(loc='best')

    plt.subplots_adjust(hspace=0.5)


def get_test_case(part_id, case_id):
    import loadTestCases
    testcase = loadTestCases.load(part_id, case_id)
    return testcase


def test_case_1():
    testcase = get_test_case(3, 1)
    engEnv = computeEngEnv(**testcase['input'])
    #plt.show()
    assert np.allclose(testcase['output'], engEnv, atol=1e-6, rtol=0)


def test_case_2():
    testcase = get_test_case(3, 2)
    engEnv = computeEngEnv(**testcase['input'])
    #plt.show()
    assert np.allclose(testcase['output'], engEnv, atol=1e-6, rtol=0)


def test_case_3():
    testcase = get_test_case(3, 3)
    engEnv = computeEngEnv(**testcase['input'])
    #plt.show()
    assert np.allclose(testcase['output'], engEnv, atol=1e-6, rtol=0)

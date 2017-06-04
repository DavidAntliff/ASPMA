import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import get_window, triang, blackmanharris
from scipy.fftpack import ifft
import os, sys
import logging

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import dftModel as DFT
import utilFunctions as UF
import sineModel as SM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# From A4Part3:
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


# From A4Part3:
def compute_eng_env(x, fs, w, M, N, H):
    mX, pX = stft.stftAnal(x, w, N, H)
    mXlinear = 10.0 ** (mX / 20.0)

    # Get an array of indices for bins within each band range:
    # Using np.where():
    bins = np.arange(0, N) * fs / N
    band_low_bins = np.where((bins > 0) & (bins < 3000.0))[0]
    band_high_bins = np.where((bins > 3000) & (bins < 10000.0))[0]

    num_frames = mX.shape[0]
    env = np.zeros(shape=(num_frames, 2))

    for frame in range(num_frames):
        env[frame, 0] = 10.0 * np.log10(sum(mXlinear[frame, band_low_bins] ** 2))
        env[frame, 1] = 10.0 * np.log10(sum(mXlinear[frame, band_high_bins] ** 2))

    plot_spectrogram_with_energy_envelope(mX, env, M, N, H, fs, 'mX, M={}, N={}, H={}'.format(M, N, H))

    return fs, mX, env


# From A4Part4:
def rectify(x):
    return x if x > 0 else 0


def computeODF(x, fs, w, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF 
            computed on the low frequency band and the second column is the ODF computed on the high 
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz 
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """

    fs, mX, env = compute_eng_env(x, fs, w, M, N, H)

    num_frames = env.shape[0]
    odf = np.zeros(shape=(num_frames, 2))

    for frame in range(1, num_frames):
        odf[frame, 0] = rectify(env[frame, 0] - env[frame - 1, 0])
        odf[frame, 1] = rectify(env[frame, 1] - env[frame - 1, 1])

    plot_spectrogram_with_odf(mX, odf, M, N, H, fs, 'mX, M={}, N={}, H={}'.format(M, N, H))

    return odf


def plot_spectrogram_with_odf(mX, odf, M, N, H, fs, title):
    assert mX.shape[0] == odf.shape[0]
    num_frames = mX.shape[0]

    frmTime = H * np.arange(num_frames) / float(fs)
    binFreq = np.arange(N / 2 + 1) * float(fs) / N

    # plt.suptitle(title)
    #
    # plt.subplot(2, 1, 1)
    # plt.title("Spectrogram")
    # plt.pcolormesh(frmTime, binFreq, np.transpose(mX), cmap='jet')
    # plt.autoscale(tight=True)
    # plt.ylim([0, 10000])
    # # plt.xlabel("Time (sec)")
    # plt.ylabel("Frequency (Hz)")

    plt.subplot(3, 1, 3)
    plt.title("Onset Detection Function")
    plt.plot(frmTime, odf[:, 0], 'b', label='Low')
    plt.plot(frmTime, odf[:, 1], 'g', label='High')
    plt.xlabel("Time (sec)")
    plt.ylabel("Magnitude (dB)")
    plt.legend(loc='best')

    # plt.subplots_adjust(hspace=0.5)


def analysis(x, fs, w, N, t):
    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    pin = int(math.floor(len(x)+1)/2)                       # init sound pointer in middle of data window

    # -----analysis-----
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
    ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
    ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz

    return iploc, ipmag, ipphase, ipfreq


def sineModelMultiRes(x, fs, w_seq, N_seq, t, B_seq):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    using multi-resolution approach.
    x: input array sound, fs: sample rate
    w: sequence of three analysis windows
    N: sequence of three sizes of complex spectrum
    t: threshold in negative dB
    B: sequence of three frequency bands, represented as (min Hz, max Hz) tuples
    returns y: output array sound
    """

    assert len(w_seq) == len(N_seq), "w_seq and N_seq must be sequences of the same size"
    assert len(w_seq) == len(B_seq), "w_seq and B_seq must be sequences of the same size"
    k = len(w_seq)

    # Each analysis frame should be the same length as the largest window
    # but each hop should be the same length as the smallest window
    min_window_size = min([item.size for item in w_seq])
    max_window_size = max([item.size for item in w_seq])
    logger.debug("min_window_size {}".format(min_window_size))
    logger.debug("max_window_size {}".format(max_window_size))

    hM1 = int(math.floor(min_window_size+1)/2)              # half analysis window size by rounding
    hM2 = int(math.floor(min_window_size/2))                # half analysis window size by floor

    hM1_max = int(math.floor(max_window_size+1)/2)
    hM2_max = int(math.floor(max_window_size/2))

    max_N = max(N_seq)

    Ns = 512                                                # FFT size for synthesis (even)
    H = Ns/4                                                # Hop size used for analysis and synthesis
    hNs = Ns/2                                              # half of synthesis FFT size
    pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window
    pend = x.size - pin                                     # last sample to start a frame
    fftbuffer = np.zeros(max_N)                             # initialize buffer for FFT
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window

    for i in range(len(w_seq)):
        w_seq[i] = w_seq[i] / sum(w_seq[i])                 # normalize analysis windows

    while pin<pend:                                         # while input sound pointer is within sound
        # -----analysis-----
        iploc = [ None ] * k
        ipmag = [ None ] * k
        ipphase = [ None ] * k
        ipfreq = [ None ] * k

        # the frame of audio to analyse must be as wide as the largest window
        x1 = x[pin-hM1_max:pin+hM2_max]
        for i, (w, N) in enumerate(zip(w_seq, N_seq)):
            iploc[i], ipmag[i], ipphase[i], ipfreq[i] = analysis(x1, fs, w, N, t)
            #import pdb; pdb.set_trace()

        # -----pick bands-----
        final_ipmag = ipmag[0]
        final_ipphase = ipphase[0]
        final_ipfreq = ipfreq[0]

        # -----synthesis-----
        #import pdb; pdb.set_trace()
        Y = UF.genSpecSines(final_ipfreq, final_ipmag, final_ipphase, Ns, fs)   # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
        yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1]
        y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
        pin += H
    return y


def energy(x):
    return sum(x ** 2)


def snr_from_energy(e_signal, e_noise):
    return 10.0 * np.log10(e_signal / e_noise)


def diff_snr(x, y):
    e_signal = energy(x)
    e_error = energy(x - y)
    return snr_from_energy(e_signal, e_error)


def main():
    inputFile = '../../sounds/orchestra.wav'
    window = 'hamming'
    t = -90

    fs, x = UF.wavread(inputFile)

    #odf = computeODF(x, fs, w, M, N, H)

    # original sineModel:
    w = get_window(window, 1023)
    y = SM.sineModel(x, fs, w, 1024, t)

    # multi-resolution sineModel:
    N1, M1, B1 = (4096, 4095, (0.0, 1000.0))
    N2, M2, B2 = (2048, 2047, (1000.0, 5000.0))
    N3, M3, B3 = (1024, 1023, (5000.0, 22050.0))

    w1 = get_window(window, M1)
    w2 = get_window(window, M2)
    w3 = get_window(window, M3)

    #y_mr = sineModelMultiRes(x, fs, [w1, w2, w3], [N1, N2, N3], t, [B1, B2, B3])
    y_mr = sineModelMultiRes(x, fs, [w3], [N3], t, [B3])

    print("SNR sineModel: {} dB".format(diff_snr(x, y)))
    print("SNR sineModelMultiRes: {} dB".format(diff_snr(x, y_mr)))
    import pdb; pdb.set_trace()

    #plt.plot(x, color='r', alpha=1.0)
    plt.plot(y, color='b', alpha=0.5)
    plt.plot(y_mr, color='g', alpha=0.5)
    #plt.plot(y - x, color='y')
    #plt.plot(y_mr - x, color='k')
    plt.show()


if __name__ == "__main__":
    main()

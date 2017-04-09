import numpy as np
import matplotlib.pyplot as plt

j = np.complex(0, 1)
pi = np.pi

def dft(N, x):
    X = np.array([])
    nv = np.arange(-N/2, N/2)
    kv = np.arange(-N/2, N/2)

    #for k in range(N):
    for k in kv:
        #s = np.exp(j * 2 * pi * k / N * np.arange(N))
        s = np.exp(j * 2 * pi * k / N * nv)
        X = np.append(X, sum(x * np.conjugate(s)))
    return X, kv

def idft(N, X):
    y = np.array([])
    nv = np.arange(-N/2, N/2)
    kv = np.arange(-N/2, N/2)
    for n in nv:
        s = np.exp(j * 2 * pi * n / N * kv)
        y = np.append(y, 1.0 / N * sum(X * s))
    return y, kv

def main_complex_exponential():
    N = 64
    k0 = 7
    x = np.exp(j * 2 * pi * k0 / N * np.arange(N))
    X, kv = dft(N, x)
    plt.plot(kv, np.abs(X))
    plt.axis([-N/2, N/2-1, 0, N])
    plt.show()

def do_real(N, k0):
    x = np.cos(2 * pi * k0 / N * np.arange(N))
    X, kv = dft(N, x)
    return X, kv

def main_real():
    N = 64
    k0 = 7.5
    X, kv = do_real(N, k0)
    #plt.plot(np.arange(N), np.abs(X))
    plt.plot(kv, np.abs(X))
    #plt.axis([0, N-1, 0, N])
    plt.axis([-N/2, N/2-1, 0, N])
    plt.show()
    return X, kv

def main_idft():
    N = 64
    k0 = 7.5
    X, _ = do_real(N, k0)
    y, kv = idft(N, X)
    plt.plot(kv, y)
    plt.axis([-N/2, N/2-1, -1, 1])
    plt.show()

if __name__ == "__main__":
    main_complex_exponential()
    main_real()
    main_idft()

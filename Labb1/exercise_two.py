import numpy as np
from numpy.fft import fft2, ifft2
from course_files.Functions import showfs, showgrey


def main():
    F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
    G = F.T
    H = F + 2*G

    Fhat = fft2(F)
    Ghat = fft2(G)
    Hhat = fft2(H)

    showgrey(np.log(1 + np.abs(Fhat)))
    showgrey(np.log(1 + np.abs(Ghat)))
    showgrey(np.log(1 + np.abs(Hhat)))
    showfs(Hhat)

    showgrey(F * G)
    showfs(fft2(F * G))

    showgrey(np.real(ifft2(np.dot(Fhat, Ghat))))
    import scipy.signal as sig
    #showgrey(ifft2(sig.convolve2d(Fhat, Ghat, 'same', boundary='fill', fillvalue=0) ))
    #showfs(sig.correlate2d(Fhat, Ghat, mode='same'))
    # Xf = Fhat.flatten()
    # Yf = Ghat.flatten()
    # N = Xf.size  # or Yf.size since they must have the same size
    # conv = np.convolve(Xf, np.concatenate((Yf, Yf)))
    # conv = conv[N:2 * N]
    # showfs(np.abs(conv.reshape((128, 128))))


if __name__ == '__main__':
    main()

import numpy as np
from course_files.Functions import *
from numpy.fft import fft2, fftshift
import math


def main():
    alpha = math.degrees(60)

    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
    G = rot(F, alpha)

    showgrey(F)
    showgrey(G)

    Fhat = fft2(F)
    Ghat = fft2(G)
    showfs(Fhat)
    showfs(Ghat)

    Hhat = rot(fftshift(Ghat), -alpha)
    showgrey(np.log(1 + abs(Hhat)))


def change_phase():
    img = np.load("../course_files/Images-npy/phonecalc128.npy")
    showgrey(img)
    re_phased = randphaseimage(img)
    showgrey(re_phased)


if __name__ == '__main__':
    main()
    #change_phase()
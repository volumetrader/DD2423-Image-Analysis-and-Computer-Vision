import numpy as np
from course_files.Functions import *
from course_files.gaussfft import gaussfft


def smoothing_one():
    office = np.load("../course_files/Images-npy/office256.npy")
    #showgrey(office)

    sap = sapnoise(office, 0.1, 255)

    reduce_add_noise()
    #reduce_sap_noise()


def reduce_add_noise():
    office = np.load("../course_files/Images-npy/office256.npy")
    add = gaussnoise(office, 16)

    showgrey(add)  # noisy picture

    #showgrey(discgaussfft(add, 0.6))
    showgrey(gaussfft(add, 1))

    #showgrey(medfilt(add,3, 3))
    #showgrey(ideal(add, 0.24))


def reduce_sap_noise():
    office = np.load("../course_files/Images-npy/office256.npy")
    sap = sapnoise(office, 0.1, 255)
    showgrey(sap)

    showgrey(discgaussfft(sap, 1))
    showgrey(gaussfft(sap, 0.8))

    showgrey(medfilt(sap, 3, 3))
    showgrey(ideal(sap, 0.29))


def smoothing_two():
    img = np.load("../course_files/Images-npy/phonecalc256.npy")
    smoothimg = img
    N = 5
    f = plt.figure()
    f.subplots_adjust(wspace=0, hspace=0)
    for i in range(N):
        #if i > 0:  # generate subsampled versions
        img = rawsubsample(img)
        smoothimg = gaussfft(smoothimg, 0.8)#
        smoothimg = rawsubsample(smoothimg)
        f.add_subplot(2, N, i + 1)
        showgrey(img, False)
        f.add_subplot(2, N, i + N + 1)
        showgrey(smoothimg, False)
    plt.show()


def main():
    reduce_add_noise()


if __name__ == '__main__':
    main()

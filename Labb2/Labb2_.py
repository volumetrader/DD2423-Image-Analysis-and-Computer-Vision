import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from course_files.Functions import *
from course_files.gaussfft import gaussfft
from course_files.Functions import showgrey

FOLDER = "../course_files/"


def deltax():
    return [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
    ]


def deltay():
    return [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
    ]


def Lv_lite(inpic, dxmask, dymask, shape='same'):
    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    return np.sqrt(Lx**2 + Ly**2)


def exercise_one(tools):
    dxtools = convolve2d(tools, deltax(), 'valid')
    dytools = convolve2d(tools, deltay(), 'valid')
    # showgrey(dxtools)
    # showgrey(dytools)
    print(f"dxtools shape: {dxtools.shape}")
    print(f"tools shape: {tools.shape}")
    showgrey(Lv_lite(tools, deltax(), deltay()))


def exercise_two(tools):
    THRESHOLD = 100
    dxtools = convolve2d(tools, deltax(), 'valid')
    dytools = convolve2d(tools, deltay(), 'valid')

    gradmagntools = np.sqrt(dxtools ** 2 + dytools ** 2)
    print(np.histogram(gradmagntools))

    showgrey((gradmagntools > THRESHOLD).astype(int))


def exercise_two_2():
    sigma = 1
    threshold = 110
    pictures = [np.load(FOLDER + i) for i in ["Images-npy/few256.npy", "Images-npy/godthem256.npy"]]
    for pic in pictures:
        smoothed_pic = discgaussfft(pic, sigma)

        #dxmask = convolve2d(smoothed_pic, deltax(), 'valid')
        #dymask = convolve2d(smoothed_pic, deltay(), 'valid')

        res = Lv_lite(smoothed_pic, deltax(), deltay())
        print(np.histogram(res))
        showgrey((res > threshold).astype(int))


def main():

    tools = np.load(FOLDER + "Images-npy/few256.npy")
    exercise_one(tools)
    exercise_two_2()


if __name__ == '__main__':
    main()

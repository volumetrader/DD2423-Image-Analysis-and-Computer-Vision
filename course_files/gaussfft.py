import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import math
from course_files.Functions import deltafcn, variance, showgrey, showfs


def gaussfft(pic, t):
    sigma_2 = t**2
    pfft = np.fft.fft2(pic)
    [h, w] = np.shape(pic)
    #h, w = h//2, w//2
    n = (1 - 1 / w) / 2
    [x, y] = np.meshgrid(np.linspace(0, (1 - 1 / w), w), np.linspace(0, (1 - 1 / h), h))
    #ffft = np.exp(t * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) - 2))
    ffft = (1/(2*np.pi*sigma_2)) * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) - 2) / (2*sigma_2))

    #ffft = (1/(2*np.pi*sigma_2)) * np.exp((np.sqrt((x-n)**2 + (y-n)**2)) / (2*sigma_2))
    #ffft = (1/(2*np.pi*sigma_2)) * (((x-n)**2 + (y-n)**2) / (2*sigma_2))

    #showfs(ffft)
    #showfs(np.exp(t * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) - 2)))

    #showfs((pfft))
    #ffft = np.exp(-((2*np.pi*x)**2 + (2*np.pi*y)**2)*(t/2))
    #ffft = np.exp(-((x)**2 + (y)**2)/(2*t))
    #ffft = (1/(2*np.pi*t)) * np.exp(-(x**2 + y**2) / (2*t))
    print(np.shape(pic), np.shape(ffft), np.shape(pfft))

    pixels = np.real(np.fft.ifft2(ffft * pfft))
    return pixels


def test_gaussfft():
    t = 0.1
    img = deltafcn(128, 128)
    #img = np.load("../course_files/Images-npy/phonecalc128.npy")

    showgrey(img)
    psf = gaussfft(img, t)
    showgrey(psf)
    var = variance(psf)
    print(f"Variance: {var}")


if __name__ == '__main__':
    test_gaussfft()

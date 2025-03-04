import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt
import time
from course_files.Functions import *
from course_files.gaussfft import gaussfft

def dx():
    return [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
    ]


def dy():
    return [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
    ]

deltay = np.array([[0.5], [0], [-0.5]])
deltax = np.array([[0.5, 0, -0.5]])
deltayy = np.array([[1], [-2], [1]])
deltaxx = np.array([[1, -2, 1]])
deltayyy = convolve2d(deltay, deltayy, mode="full")  # valid
deltaxxx = convolve2d(deltax, deltaxx, mode="full") # 'valid
deltaxy = convolve2d(deltax, deltay, mode="full")
deltaxxy = convolve2d(deltaxx, deltay, mode="full")
deltaxyy = convolve2d(deltax, deltayy, mode="full")


def Ly(inpic, shape='valid'):
    return convolve2d(inpic, deltay, shape)


def Lx(inpic, shape='valid'):
    return convolve2d(inpic, deltax, shape)


def Lyy(inpic, shape='valid'):
    return convolve2d(inpic, deltayy, shape)


def Lxx(inpic, shape='valid'):
    return convolve2d(inpic, deltaxx, shape)


def Lyyy(inpic, shape='valid'):
    return convolve2d(inpic, deltayyy, shape)


def Lxxx(inpic, shape='valid'):
    return convolve2d(inpic, deltaxxx, shape)


def Lxy(inpic, shape='valid'):
    return convolve2d(inpic, deltaxy, shape)


def Lxyy(inpic, shape='valid'):
    return convolve2d(inpic, deltaxyy, shape)


def Lxxy(inpic, shape='valid'):
    return convolve2d(inpic, deltaxxy, shape)


def Lv(inpic, shape='valid'):
    Lx = convolve2d(inpic, dx(), shape)
    Ly = convolve2d(inpic, dy(), shape)
    return np.sqrt(Lx ** 2 + Ly ** 2)


def Lvvtilde(inpic, shape='valid'):
    return (
            Lx(inpic, shape)**2*Lxx(inpic, shape) + 2*Lx(inpic, shape) * Ly(inpic, shape) * Lxy(inpic, shape) +
            Ly(inpic, shape)**2 * Lyy(inpic, shape)
    )


def Lvvvtilde(inpic, shape='valid'):
    return (
            Lx(inpic, shape)**3 * Lxxx(inpic, shape) + 3*Lx(inpic, shape)**2 * Ly(inpic, shape) * Lxxy(inpic, shape) +
            3*Lx(inpic, shape) * Ly(inpic, shape)**2 * Lxyy(inpic, shape) + Ly(inpic, shape)**3 * Lyyy(inpic, shape)
    )


def extractedge(inpic, scale, shape, threshold=None):
    smoothed = discgaussfft(inpic, scale)

    vv = contour(Lvvtilde(smoothed, shape)) - 1
    vvv = (Lvvvtilde(smoothed, shape) < 0).astype(int)

    curves = zerocrosscurves(vv, vvv)
    if threshold is not None:
        thres_mask = Lv(smoothed, shape)
        thres_mask = thres_mask > threshold
        curves = thresholdcurves(curves, thres_mask)
    return curves


def houghline(curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False):
    diag_len = int(max([math.sqrt(x ** 2 + y ** 2) for x, y in zip(*curves)]))
    rhos = np.linspace(-diag_len, diag_len, nrho)
    theta = np.linspace(-np.pi/2, np.pi/2, ntheta)
    acc = np.zeros((2*nrho, ntheta))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    for x, y in zip(*curves):
        rho = ((diag_len + (cos_theta * x + sin_theta * y.T)) / (2*diag_len/nrho)).astype(int)
        acc[rho, range(ntheta)] += 1 + 0.05 * np.sqrt(magnitude[x-1, y-1])

    pos, value, _ = locmax8(acc)
    indexvector = np.argsort(value)[-nlines:]
    pos = pos[indexvector]

    linepar = list()
    for idx in range(nlines):
        thetaidxacc = pos[idx, 0]
        rhoidxacc = pos[idx, 1]
        print(f"rhoidxacc: {rhoidxacc}, thetaidxacc: {thetaidxacc}")
        linepar.append((rhos[rhoidxacc], theta[thetaidxacc]))

    if verbose:
        pixel_max = int(max([max(i, j) for i, j in linepar]))
        plot_lines(np.zeros((pixel_max, pixel_max)), linepar)
    return linepar, acc


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
    print(f"matrix is {nrho*ntheta} nubmer of cells.")
    t1 = time.time()
    THRESHOLD = 20
    curves = extractedge(pic, scale=scale, shape="same", threshold=THRESHOLD)
    overlaycurves(pic, curves)
    linepar, acc = houghline(curves, gradmagnthreshold, nrho, ntheta, threshold=THRESHOLD, nlines=nlines, verbose=verbose)
    print(f"Time to compute line was {time.time() - t1}")
    return linepar, acc


def plot_lines(image, lines):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    from matplotlib.lines import Line2D

    for rho, theta in lines:
        b = np.cos(theta)
        a = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        print(f"x0: {x0}, y0: {y0}")
        line = Line2D([x1, x2], [y1, y2], color='red')
        ax.add_line(line)

    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([image.shape[1], 0])  # Invert y-axis to match image coordinates

    plt.title('Hough Lines')
    plt.show()


def question_four():
    scale = 4
    house = np.load("../course_files/Images-npy/godthem256.npy")
    print(Lvvtilde(discgaussfft(house, scale), 'same'))
    showgrey(contour(Lvvtilde(discgaussfft(house, scale), 'same')))


def question_five():
    scale = 4
    tools = np.load("../course_files/Images-npy/few256.npy")
    showgrey(contour(Lvvtilde(discgaussfft(tools, scale), 'same')))
    showgrey((Lvvvtilde(discgaussfft(tools, scale), 'same') < 0).astype(int))
    curves = extractedge(tools, scale, "same")
    overlaycurves(tools, curves)
    plt.show()


def question_seven():
    scale = 9
    threshold = 30
    house = np.load("../course_files/Images-npy/godthem256.npy")
    tools = np.load("../course_files/Images-npy/few256.npy")
    #for image in [house, tools]:
    image = tools
    #smoothed_image = discgaussfft(image, scale)
    curves = extractedge(image, scale, "same", threshold)
    overlaycurves(image, curves)
    plt.show()


def main():
    [x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
    scale = 8

    house = np.load("../course_files/Images-npy/houghtest256.npy")

    mag_threshold = Lv(house)
    linepar, acc = houghedgeline(
        house,
        scale=scale,
        gradmagnthreshold=mag_threshold,
        nrho=500,
        ntheta=300,
        nlines=3,
        verbose=False
    )
    plot_lines(house, linepar)
    #overlaycurves(house, curves)
    plt.show()


if __name__ == '__main__':
    main()

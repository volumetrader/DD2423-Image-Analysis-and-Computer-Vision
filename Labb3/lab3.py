import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from course_files.Functions import *
from course_files.gaussfft import gaussfft
from PIL import Image


def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """
    # Initialization
    #pixels = np.asarray(image).astype(np.ubyte)
    N = 10
    original_shape = image.shape
    pixels = np.reshape(image, (-1, 3))
    np.random.seed(seed)
    idxs = np.random.randint(0, pixels.shape[0], K*N)
    r_values = pixels[idxs]
    r_idxs = np.argsort(distance_matrix(r_values, [(0, 0, 0)]))
    r_values = r_values[r_idxs]

    centers = list()

    for i in range(K):
        center = np.mean(r_values[i*N:(i+1)*N].reshape(-1, 3), axis=0)
        #print(f"center: {center}")
        centers.append(center)
    #centers[0] = np.array([0, 0, 0])
    #centers[K-1] = np.array([256, 256, 256])
    #centers = np.array(centers)

    segmentation = np.zeros((pixels.shape[0], 1))
    distances = distance_matrix(pixels, centers)
    # for c in clusters:
    #     distances = np.sqrt(np.sum(np.square(pixels[:]-c), axis=1))
    #     print(np.sqrt(np.sum(np.square(pixels[:]-c), axis=1)))
    #print(distances[:10])
    for t in range(L):
        for i, d in enumerate(distances):
            segmentation[i] = np.argmin(d).astype(np.int32)
        classes = len(np.unique(segmentation))
        #print(f"classes: {classes}")
        for c in range(K):
            idxs = np.argwhere(segmentation == c)
            if len(idxs) == 0:
                continue
            #print(f"idxs: {len(idxs)}")
            centers[c] = np.mean(pixels[idxs, :], axis=0)[0].astype(np.int32).astype(int)

        #centers = [c for c in centers if c.sum() != 0]
        distances = distance_matrix(pixels, centers)


    segmentation = np.resize(segmentation.astype(int), (original_shape[0], original_shape[1]))
    return segmentation, np.array(centers)  #[c for c in centers if c.sum() != 0]


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """
    r_image = np.reshape(image, (-1, 3)).astype(np.float32)
    r_mask = np.reshape(mask, (-1)).astype(bool)
    masked_img = r_image[r_mask, :]  #np.reshape(masked_img, (-1, 3)).astype(np.float32)
    N = len(masked_img)

    # Randomly initialize the K components using masked pixels
    seg, centers = kmeans_segm(image, K, 40, seed=24)
    probs = np.ones((K, N)) / N
    weights = np.array([np.count_nonzero(seg == i) for i in range(K)]) / len(r_image)
    #print(f"weights: {weights}")
    means = centers #np.ones((K, 3)) * np.mean(masked_img, axis=0)
    covariace = np.array([(i+1)*np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]]) for i in range(K)])# np.ones((K, 3, 3)) * 20
    #covariace = np.ones((K, 3, 3)) * 20
    # for k_idx in range(K):
    #     res = np.dot((masked_img - means[k_idx]).T, (masked_img - means[k_idx]))
    #     covariace[k_idx] = res

    def gaussian_function(ci, mu_k, sigma_k):
        """
        Compute the value of the multivariate Gaussian function.

        Parameters:
        - ci: Input matrix (numpy array of shape (N, 3))
        - mu_k: Mean vector (numpy array of shape (3,))
        - sigma_k: Covariance matrix (numpy array of shape (3, 3))

        Returns:
        - Value of the Gaussian function for the given input matrix and parameters
        """
        dim = len(mu_k)
        if np.linalg.det(sigma_k) < 1e-2:
            sigma_k = sigma_k / 10000
            epsilon = 1e-0
            sigma_k = sigma_k.astype(np.float32)
            sigma_k += epsilon * np.eye(sigma_k.shape[0])
            print(f"sigma_k:", sigma_k)

        norm_sigma_k = np.linalg.det(sigma_k)
        try:
            inv_sigma_k = np.linalg.inv(sigma_k)
        except Exception:
            inv_sigma_k = 0

        mu_k = mu_k.reshape(-1, 1)
        # Compute the exponent term for each row in ci
        diff = ci - mu_k.T  # Subtract mu_k and transpose to make broadcasting work
        return (1 / np.sqrt((2 * np.pi) ** 3 * norm_sigma_k)) * np.exp(-0.5 * np.sum(diff * np.dot(inv_sigma_k, diff.T).T, axis=1))


    def update_covariance(ci, mu_k, pi_k):
        r = np.dot((pi_k[:, None] * (ci-mu_k)).T, (ci-mu_k))
        return r / np.sum(pi_k)

    g = gaussian_function(masked_img, means[0], covariace[0])

    for i in range(L):
        # Expectation: Compute probabilities P_ik using masked pixels
        p_sum = 0
        for j in range(K):
            w_prob = weights[j] * gaussian_function(masked_img, means[j], covariace[j])
            p_sum += w_prob
            probs[j] = w_prob
        probs /= p_sum

        #Maximization: Update weights, means and covariances using masked pixels
        weights = np.mean(probs, axis=1)
        for j in range(K):
            if np.sum(probs[j]) == 0:
                means[j] = np.sum(np.dot(probs[j], masked_img)) / np.sum(probs[j])
            else:
                means[j] = 0
        for j in range(K):
            covariace[j] = update_covariance(masked_img, means[j], probs[j])

    # Compute probabilities p(c_i) in Eq.(3) for all pixels I.
    gass = np.zeros((K, r_image.shape[0]))
    #means = centers
    p_sum = 0
    for j in range(K):
        res = gaussian_function(r_image, means[j], covariace[j]) * weights[j]
        gass[j] = res
        p_sum += res
    prob = np.sum(gass.T / weights, axis=1)
    prob = prob.reshape((image.shape[0], image.shape[1]))
    return prob


def main():
    K = 10
    L = 20
    image = Image.open('../course_files/Images-jpg/orange.jpg')

    image = np.asarray(image).astype(np.float32)
    kmeans_segm(image=image, K=K, L=L)
    #kmeans_example()


if __name__ == '__main__':
    main()

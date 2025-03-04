from PIL import Image
import numpy as np


img = Image.open('../course_files/Images-jpg/orange.jpg')

I = np.asarray(img).astype(np.float32)
print(I)

Ivec = np.reshape(I, (-1, 3))
#print(Ivec[100])

"""
Let X be a set of pixels and V be a set of K cluster centers in 3D (R,G,B).
% Randomly initialize the K cluster centers
% Compute all distances between pixels and cluster centers
% Iterate L times
% Assign each pixel to the cluster center for which the distance is minimum
% Recompute each cluster center by taking the mean of all pixels assigned to it
% Recompute all distances between pixels and cluster centers
"""

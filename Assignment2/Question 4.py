import numpy as np
from skimage.feature import graycomatrix, graycoprops
import math

# Given image
image = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 2, 2, 2],
    [2, 2, 3, 3]
])

# Compute and normalize co-occurrence matrix
def compute_co_matrix(img, distances, angles):
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=4, symmetric=True, normed=False)
    return glcm[:, :, 0, 0]

# a) Compute requested matrices
P_45_1 = compute_co_matrix(image, distances=[1], angles=[np.pi/4])
P_90_1 = compute_co_matrix(image, distances=[1], angles=[np.pi/2])
P_135_1 = compute_co_matrix(image, distances=[1], angles=[3*np.pi/4])
P_0_2 = compute_co_matrix(image, distances=[2], angles=[0])

print("P_45_1:\n", P_45_1)
print("\nP_90_1:\n", P_90_1)
print("\nP_135_1:\n", P_135_1)
print("\nP_0_2:\n", P_0_2)

# b) Calculate texture features for P_{0°,1}
P_0_1 = np.array([
    [4, 2, 1, 0],
    [2, 4, 0, 0],
    [1, 0, 6, 1],
    [0, 0, 1, 2]
])

# Normalize the matrix
P_norm = P_0_1 / np.sum(P_0_1)

# i) Energy
energy = np.sum(P_norm**2)

# ii) Entropy
entropy = -np.sum(P_norm[P_norm > 0] * np.log2(P_norm[P_norm > 0]))

# iii) Maximum probability
max_prob = np.max(P_norm)

# iv) Contrast
contrast = 0
for i in range(P_norm.shape[0]):
    for j in range(P_norm.shape[1]):
        contrast += (i - j)**2 * P_norm[i, j]

# v) Inverse Difference Moment
idm = 0
for i in range(P_norm.shape[0]):
    for j in range(P_norm.shape[1]):
        idm += P_norm[i, j] / (1 + (i - j)**2)

# vi) Correlation
mean_i = np.sum([i * np.sum(P_norm[i, :]) for i in range(P_norm.shape[0])])
mean_j = np.sum([j * np.sum(P_norm[:, j]) for j in range(P_norm.shape[1])])
std_i = np.sqrt(np.sum([(i - mean_i)**2 * np.sum(P_norm[i, :]) for i in range(P_norm.shape[0])]))
std_j = np.sqrt(np.sum([(j - mean_j)**2 * np.sum(P_norm[:, j]) for j in range(P_norm.shape[1])]))

correlation = 0
for i in range(P_norm.shape[0]):
    for j in range(P_norm.shape[1]):
        correlation += ((i - mean_i) * (j - mean_j) * P_norm[i, j]) / (std_i * std_j)

print("\nEnergy:", energy)
print("Entropy:", entropy)
print("Maximum Probability:", max_prob)
print("Contrast:", contrast)
print("Inverse Difference Moment:", idm)
print("Correlation:", correlation)

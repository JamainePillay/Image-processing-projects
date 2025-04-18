import numpy as np
from collections import Counter

# Data
data = {
    'ω1': [(4, 4), (14, 7), (18, 6), (7, 4), (14, 9), (16, 2), (2, 16)],
    'ω2': [(-7, 8), (3, 11), (-2, -1), (4, 3), (3, 4), (4, -3), (0, 6)],
    'ω3': [(3, 2), (8, 1), (3, 3), (6, 3), (5, 10), (3, 9), (1, 7)]
}

# Test
test_sample = np.array([4, 3])

# Compute distances
distances = []
for cls, points in data.items():
    for point in points:
        dist = np.linalg.norm(test_sample - np.array(point))
        distances.append((dist, cls, point))

# Sort by distance
distances.sort()

# Classify based on k nearest neighbors
def classify_k_nearest(k):
    nearest = distances[:k]
    labels = [cls for _, cls, _ in nearest]
    return Counter(labels).most_common(1)[0][0], nearest

# a) Nearest neighbour (k=1)
class_1nn, nn_1 = classify_k_nearest(1)
print("a) Nearest neighbour Rule (k=1):")
print(f"   → Class: {class_1nn}")
print(f"   → Nearest point: {nn_1}\n")

# b) 3-nearest neighbour (k=3)
class_3nn, nn_3 = classify_k_nearest(3)
print("b) 3-Nearest neighbour Rule (k=3):")
print(f"   → Class: {class_3nn}")
print(f"   → Nearest points: {nn_3}\n")

# c) 7-nearest neighbour (k=7)
class_7nn, nn_7 = classify_k_nearest(7)
print("c) 7-Nearest neighbour Rule (k=7):")
print(f"   → Class: {class_7nn}")
print(f"   → Nearest points: {nn_7}")

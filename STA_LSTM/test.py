import numpy as np
from scipy.spatial import procrustes

# Two sets of 3D points
source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
target_points = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

# Estimate rotation matrix using Procrustes analysis
x, y, rotation = procrustes(source_points, target_points)

print("Rotation matrix:")
print(rotation)

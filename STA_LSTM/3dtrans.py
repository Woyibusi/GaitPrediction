import numpy as np
from scipy.linalg import orthogonal_procrustes

# Two sets of 3D points
source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
target_points = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

# Estimate rigid transformation using Procrustes analysis
rotation, translation= orthogonal_procrustes(source_points, target_points)

# Print the estimated transformation
print("Rotation matrix:")
print(rotation)
print("Translation vector:")
print(translation)


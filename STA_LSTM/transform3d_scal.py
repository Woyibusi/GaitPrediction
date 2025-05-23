import numpy as np
from scipy.linalg import svd
import math
from scipy.spatial import distance
from scipy.linalg import orthogonal_procrustes


def est_similarity_trans(source_points):
    # source points are the corrdinates of facial keypoints 226  (left eye corner),446(right eye corner),2 (below nose) (using mediapipe)
    target_points = [[0.60, 0.57, 0.00], [0.44, 0.57, 0.00], [0.52, 0.70, 0]]
    ave_dist_source = (distance.euclidean(source_points[0],source_points[1])+distance.euclidean(source_points[0],source_points[2])+distance.euclidean(source_points[1],source_points[2]))/3
    ave_dist_target = 0.15509333
    scale_factor= ave_dist_target/ave_dist_source

    source_points *= scale_factor
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_points_centered = source_points - centroid_source
    target_points_centered = target_points - centroid_target
    # Calculate the scaling factor
    #R, sca = orthogonal_procrustes(source_points_centered , target_points_centered)
    #x, y, R = procrustes(source_points_centered , target_points_centered)

    U, _, Vt = svd(source_points_centered.T.dot(target_points_centered))  # Compute the singular value decomposition (SVD) of the matrix
    R = Vt.T.dot(U.T)  # Compute the rotation matrix R as the product of Vt.T.dot(U.T):
    t = centroid_target - R.dot(centroid_source)  # Compute the translation vector t
    return R, t, scale_factor


def similarity_trans(source_points, R, t,scale_factor):
    transformed_source_points = scale_factor * np.dot(R, source_points.T).T + t
    return transformed_source_points


"""
source_points = np.array([[ 0.44, 0.57,  0.00],[ 0.60,  0.57,  0.00],[ 0.52,  0.66, 0]])
target_points = np.array([[ 0.44, 0.57,  0.00],[ 0.60,  0.57,  0.00],[ 0.52,  0.66, 0]])
R, t = est_similarity_trans(source_points)
transformed_source_points = np.dot(R, source_points.T).T + t
print(transformed_source_points)
print(target_points)
"""



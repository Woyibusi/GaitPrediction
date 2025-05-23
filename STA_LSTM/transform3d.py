import numpy as np
from scipy.linalg import svd
import math

def est_similarity_trans(source_points):
    #source points are the corrdinates of facial keypoints 226  (left eye corner),446(right eye corner),2 (below nose) (using mediapipe)
    target_points=[[ 0.60, 0.57,  0.00],[ 0.44,  0.57,  0.00],[ 0.52,  0.70, 0]]

    # computer mean
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # shift by mean
    source_points_centered = source_points - centroid_source
    target_points_centered = target_points - centroid_target

    H = source_points_centered@ np.transpose(target_points_centered )

    # Find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T


    #U, _, Vt = svd(source_points_centered.T.dot(target_points_centered))  # Compute the singular value decomposition (SVD) of the matrix
    #R = Vt.T.dot(U.T)  # Compute the rotation matrix R as the product of Vt.T.dot(U.T):
    t = centroid_target - R.dot(centroid_source)  # Compute the translation vector t
    return R,t

def similarity_trans(source_points,R,t):
    transformed_source_points = np.dot(R, source_points.T).T + t
    return transformed_source_points


"""
source_points = np.array([[ 0.44, 0.57,  0.00],[ 0.60,  0.57,  0.00],[ 0.52,  0.66, 0]])
target_points = np.array([[ 0.44, 0.57,  0.00],[ 0.60,  0.57,  0.00],[ 0.52,  0.66, 0]])
R, t = est_similarity_trans(source_points)
transformed_source_points = np.dot(R, source_points.T).T + t
print(transformed_source_points)
print(target_points)
"""



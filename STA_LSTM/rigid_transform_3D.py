#!/usr/bin/python

import numpy as np
from scipy.spatial import distance

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector


def est_similarity_trans(source_points):
    # source points are the corrdinates of facial keypoints 226  (left eye corner),446(right eye corner),2 (below nose) (using mediapipe)
    target_points= np.array([[0.60, 0.57, 0.00], [0.44, 0.57, 0.00], [0.52, 0.70, 0]])

    assert source_points.shape == source_points.shape

    num_rows, num_cols = source_points.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = target_points.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(source_points, axis=0)
    centroid_B = np.mean(target_points, axis=0)


    # subtract mean
    source_points = source_points - centroid_A
    target_points = target_points - centroid_B

    # Calculate the scaling factor
    #scale_factor = np.linalg.norm(Bm) / np.linalg.norm(Am)
    ave_dist_source = (distance.euclidean(source_points[0], source_points[1]) + distance.euclidean(source_points[0],source_points[2]) + distance.euclidean(source_points[1], source_points[2])) / 3
    ave_dist_target = 0.15509333
    scale_factor = ave_dist_target / ave_dist_source
    # Scale the source point cloud
    source_points *= scale_factor

    # Compute the covariance matrix
    H = source_points @ np.transpose(source_points)

    # Find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = -R @ centroid_A * scale_factor + centroid_B

    return R, t, scale_factor


def similarity_trans(source_points, R, t, scale_factor):

    for i in range(len(source_points)):
        source_points[i,:]=scale_factor * (R@source_points[i,:]) + t
    #transformed_source_points = scale_factor * np.dot(R, source_points.T).T + t
    return source_points

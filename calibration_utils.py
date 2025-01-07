# Credit to Georgia Tech
import numpy as np
from random import sample

def calculate_projection_matrix(Points_2D, Points_3D):
    """Computes the 3x4 projection matrix given corresponding 2D and 3D points."""
    assert Points_2D.shape[0] == Points_3D.shape[0], "Mismatch in number of 2D-3D points."

    # Convert 3D points to homogeneous coordinates by appending a column of ones
    wp = np.column_stack((Points_3D, [1] * Points_3D.shape[0]))

    # Construct the first part of matrix A: [X Y Z 1  0  0  0  0]
    wp1 = np.concatenate((wp, np.zeros_like(wp)), axis=1).reshape((-1, 4))

    # Construct the second part of matrix A: [0  0  0  0  X Y Z 1]
    wp2 = np.concatenate((np.zeros_like(wp), wp), axis=1).reshape((-1, 4))

    # Construct the third part of matrix A: [-uX -uY -uZ -u  -vX -vY -vZ -v]
    wp3 = -np.multiply(Points_2D.reshape((-1, 1)).repeat(3, axis=1), Points_3D.repeat(2, axis=0))

    # Combine all parts to form the final A matrix for solving the least squares problem
    A = np.concatenate((wp1, wp2, wp3), axis=1)

    # Flatten the 2D point array into a column vector for solving Ax = b
    b = Points_2D.reshape((-1, 1))

    # Solve for the projection matrix M using least squares
    sol = np.linalg.lstsq(A, b, rcond=None)[0]

    # Append 1 to maintain the homogeneous coordinate and reshape into 3x4 matrix
    M = np.append(sol, [1]).reshape((3, 4))


    return M


def compute_camera_center(M):
    """Computes the camera center in world coordinates from a 3x4 projection matrix."""
    Q, m = M[:, :3], M[:, 3]
    Center = -np.linalg.pinv(Q) @ m
    return Center


def estimate_fundamental_matrix(Points_a, Points_b):
    """Computes the 3x3 fundamental matrix given corresponding 2D points from two images with rank-2 enforcement."""
    assert Points_a.shape[0] == Points_b.shape[0]

    Pa = np.hstack((Points_a, np.ones((Points_a.shape[0], 1))))
    Pb = np.hstack((Points_b, np.ones((Points_b.shape[0], 1))))
    
    A = (Pa[:, :, None] * Pb[:, None, :]).reshape(-1, 9)
    
    _, _, V = np.linalg.svd(A)
    F_matrix = V[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diag(S) @ Vt
    
    return F_matrix

def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """Randomly perturbs a fraction of points within a given interval while ensuring boundary constraints.

    Args:
        points (np.ndarray): Array of shape (num_points, 2) containing <x, y> coordinates.
        h (int): Image height, ensuring points remain within [0, h].
        w (int): Image width, ensuring points remain within [0, w].
        interval (int, optional): Range for perturbation, i.e., if interval=3, noise is sampled from [-3, 3]. Defaults to 3.
        ratio (float, optional): Proportion of points to perturb, e.g., 0.2 perturbs 20% of the points. Defaults to 0.2.

    Returns:
        np.ndarray: Adjusted points with noise applied, clipped to image boundaries.
    """
    num_noise = int(points.shape[0] * ratio)
    noise = np.zeros_like(points)
    noise[:num_noise] = np.random.uniform(-interval, interval, (num_noise, 2))
    
    np.random.shuffle(noise)  # Randomly distribute noise
    return np.clip(points + noise, [0, 0], [w, h])

def apply_matching_noise(points, ratio=0.2):
    """Randomly shuffles a fraction of points to introduce matching noise.

    Args:
        points (np.ndarray): Array of shape (num_points, 2) containing <x, y> coordinates.
        ratio (float, optional): Proportion of points to shuffle. Defaults to 0.2.

    Returns:
        np.ndarray: Points with randomly shuffled matches.
    """
    num_noise = int(len(points) * ratio)
    shuffled_indices = np.random.choice(len(points), num_noise, replace=False)

    shuffled_points = points[shuffled_indices].copy()
    np.random.shuffle(shuffled_points)

    points[shuffled_indices] = shuffled_points
    return points


def ransac_fundamental_matrix(matches_a, matches_b):
    """Estimates the fundamental matrix using RANSAC to filter outliers.

    Args:
        matches_a (np.ndarray): Nx2 array of corresponding points in Image A.
        matches_b (np.ndarray): Nx2 array of corresponding points in Image B.

    Returns:
        np.ndarray: Best 3x3 fundamental matrix.
        np.ndarray: Inlier points from Image A.
        np.ndarray: Inlier points from Image B.
    """

    # RANSAC parameters
    n = 9          # Number of points per sample
    thresh = 0.005 # Error threshold for inliers
    normalize = True

    # --------------------------- Step 1: Normalize Points (if enabled) ---------------------------
    if normalize:
        matches_a_norm = np.zeros_like(matches_a)
        matches_b_norm = np.zeros_like(matches_b)

        mean_a, mean_b = np.mean(matches_a, axis=0), np.mean(matches_b, axis=0)
        std_a, std_b = np.std(matches_a - mean_a, axis=0), np.std(matches_b - mean_b, axis=0)

        # Compute scaling transformation matrices
        S_a, S_b = np.eye(3), np.eye(3)
        S_a[0, 0], S_a[1, 1] = 1.0 / std_a[0], 1.0 / std_a[1]
        S_b[0, 0], S_b[1, 1] = 1.0 / std_b[0], 1.0 / std_b[1]

        # Compute translation transformation matrices
        O_a, O_b = np.eye(3), np.eye(3)
        O_a[0, 2], O_a[1, 2] = -mean_a[0], -mean_a[1]
        O_b[0, 2], O_b[1, 2] = -mean_b[0], -mean_b[1]

        # Compute final normalization transformation
        T_a, T_b = S_a @ O_a, S_b @ O_b

        # Apply normalization to all points
        for i in range(matches_a.shape[0]):
            matches_a_norm[i] = (T_a @ np.append(matches_a[i], [1]))[:2]
            matches_b_norm[i] = (T_b @ np.append(matches_b[i], [1]))[:2]

    else:
        matches_a_norm, matches_b_norm = matches_a.copy(), matches_b.copy()
        T_a, T_b = np.eye(3), np.eye(3)

    # --------------------------- Step 2: RANSAC to Find Best Fundamental Matrix ---------------------------
    while True:
        # Randomly sample n point pairs
        idx = np.random.choice(matches_b.shape[0], n, replace=False)
        points_a, points_b = matches_a[idx, :], matches_b[idx, :]
        points_a_norm, points_b_norm = matches_a_norm[idx, :], matches_b_norm[idx, :]

        # Compute candidate fundamental matrix
        F_candidate = estimate_fundamental_matrix(points_a_norm, points_b_norm)
        F_candidate = T_b.T @ F_candidate @ T_a  # De-normalize

        # Compute errors for all points
        inliers = []
        for j in range(matches_a.shape[0]):
            a = np.append(matches_b[j], [1]) @ F_candidate  # Compute epipolar constraint
            b = np.append(matches_a[j], [1])
            error = np.abs(a @ b)

            if error < thresh:
                inliers.append(j)

        # If enough inliers are found, break out of RANSAC loop
        if len(inliers) >= int(0.2 * matches_a.shape[0]):
            break

    # --------------------------- Step 3: Compute Best Fundamental Matrix Using Inliers ---------------------------
    Best_Fmatrix = estimate_fundamental_matrix(matches_a_norm[inliers], matches_b_norm[inliers])
    Best_Fmatrix = T_b.T @ Best_Fmatrix @ T_a  # De-normalize

    print("Fundamental matrix:\n", Best_Fmatrix)

    # Extract up to 30 inliers for visualization
    inliers_a = np.array([matches_a[i] for i in inliers[:30]])
    inliers_b = np.array([matches_b[i] for i in inliers[:30]])

    return Best_Fmatrix, inliers_a, inliers_b


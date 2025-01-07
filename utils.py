# Credit to Georgia Tech
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from scipy import io
from skimage import img_as_float32, transform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib; matplotlib.use('TkAgg')

# Constants
ORB_NUM_POINTS = 3000

def evaluate_points(M, Points_2D, Points_3D):
    """Projects 3D points using the projection matrix and computes residual error."""
    reshaped_points = np.hstack((Points_3D, np.ones((Points_3D.shape[0], 1))))
    Projection = (M @ reshaped_points.T).T
    u, v = Projection[:, 0] / Projection[:, 2], Projection[:, 1] / Projection[:, 2]
    Residual = np.sum(np.sqrt((u - Points_2D[:, 0])**2 + (v - Points_2D[:, 1])**2))
    return np.vstack([u, v]).T, Residual

def visualize_points(Actual_Pts, Projected_Pts):
    """Displays actual and projected 2D points."""
    plt.scatter(Actual_Pts[:, 0], Actual_Pts[:, 1], marker='o', label='Actual Points')
    plt.scatter(Projected_Pts[:, 0], Projected_Pts[:, 1], marker='x', label='Projected Points')
    plt.legend()
    plt.show()

def plot3dview(Points_3D, camera_center):
    """Plots 3D points and the estimated camera center."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*Points_3D.T, c='b', marker='o', label='3D Points')
    
    min_z = np.min(Points_3D[:, 2])
    for point in Points_3D:
        ax.plot([point[0], point[0]], [point[1], point[1]], [point[2], min_z], c='gray')

    if camera_center is not None:
        ax.scatter(*camera_center, s=100, c='r', marker='x', label='Camera Center')
        ax.plot([camera_center[0]] * 2, [camera_center[1]] * 2, [camera_center[2], min_z], c='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=31, azim=-129)
    plt.legend()
    plt.show()

def draw_epipolar_lines(F_matrix, ImgLeft, ImgRight, PtsLeft, PtsRight):
    """Draws epipolar lines on left and right images based on the fundamental matrix."""
    plt.figure(figsize=(10, 5))

    for idx, (img, pts, is_left) in enumerate([(ImgRight, PtsLeft, False), (ImgLeft, PtsRight, True)]):
        plt.subplot(1, 2, idx + 1)
        plt.imshow(img)
        plt.axis('off')

        for pt in pts:
            e = (F_matrix @ np.append(pt, 1)) if is_left else (F_matrix.T @ np.append(pt, 1))
            x = np.array([1, img.shape[1]])
            y = - (e[0] * x + e[2]) / e[1]
            plt.plot(x, y, c='b', linewidth=0.5)

        plt.scatter(pts[:, 0], pts[:, 1], c='r', marker='o', s=10)

    plt.show()

def get_ground_truth(eval_file, scale_factor_A=1, scale_factor_B=1):
    """Loads ground truth 2D correspondences from a .mat file."""
    file_contents = io.loadmat(eval_file)
    matches_A = np.hstack((file_contents['x1'] * scale_factor_A, file_contents['y1'] * scale_factor_A))
    matches_B = np.hstack((file_contents['x2'] * scale_factor_B, file_contents['y2'] * scale_factor_B))
    return matches_A, matches_B

def matchAndShowCorrespondence(imgA, imgB):
    """Finds and visualizes ORB feature correspondences."""
    orb = cv2.ORB_create(nfeatures=ORB_NUM_POINTS)
    kp1, des1 = orb.detectAndCompute(imgA, None)
    kp2, des2 = orb.detectAndCompute(imgB, None)

    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2), key=lambda x: x.distance)
    img3 = cv2.drawMatches(imgA, kp1, imgB, kp2, matches, None, flags=2)

    matches_kp1 = np.array([kp1[m.queryIdx].pt for m in matches])
    matches_kp2 = np.array([kp2[m.trainIdx].pt for m in matches])

    CombineReduce = np.unique(np.hstack((matches_kp1, matches_kp2)), axis=0)
    matches_kp1, matches_kp2 = CombineReduce[:, :2], CombineReduce[:, 2:]

    plt.imshow(img3)
    plt.axis('off')
    plt.show()

    save_path = os.path.join(os.path.dirname(__file__), 'vis_arrows.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Saving visualization to {save_path}\n')

    return matches_kp1, matches_kp2


def showCorrespondence(imgA, imgB, matches_kp1, matches_kp2):

    imgA = img_as_float32(imgA)
    imgB = img_as_float32(imgB)

    fig = plt.figure()
    plt.axis('off')

    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    numColors = imgA.shape[2]

    newImg = np.zeros([Height, Width, numColors])
    newImg[0:imgA.shape[0], 0:imgA.shape[1], :] = imgA
    newImg[0:imgB.shape[0], -imgB.shape[1]:, :] = imgB
    plt.imshow(newImg)

    shift = imgA.shape[1]
    for i in range(0, matches_kp1.shape[0]):

        r = lambda: random.randint(0, 255)
        cur_color = ('#%02X%02X%02X' % (r(), r(), r()))

        x1 = matches_kp1[i, 1]
        y1 = matches_kp1[i, 0]
        x2 = matches_kp2[i, 1]
        y2 = matches_kp2[i, 0]

        x = np.array([x1, x2])
        y = np.array([y1, y2 + shift])
        plt.plot(y, x, c=cur_color, linewidth=0.5)

    plt.show()


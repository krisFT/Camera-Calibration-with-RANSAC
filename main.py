# Credit to Georgia Tech
import warnings
import numpy as np
import os
import cv2
import argparse
from skimage import io, transform
from calibration_utils import (
    calculate_projection_matrix, compute_camera_center, 
    estimate_fundamental_matrix, ransac_fundamental_matrix, 
    apply_positional_noise, apply_matching_noise
)
from utils import (
    evaluate_points, visualize_points, plot3dview, 
    draw_epipolar_lines, matchAndShowCorrespondence, 
    showCorrespondence, get_ground_truth
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    data_dir = os.path.join(os.path.dirname(__file__), './data/')
    print("Reading data from: ", data_dir)
    
    Points_2D = np.loadtxt(os.path.join(data_dir, 'pts2d-norm-pic_a.txt'))
    Points_3D = np.loadtxt(os.path.join(data_dir, 'pts3d-norm.txt'))
    
    if args.hard_points:
        Points_2D = np.loadtxt(os.path.join(data_dir, 'pts2d-pic_b.txt'))
        Points_3D = np.loadtxt(os.path.join(data_dir, 'pts3d.txt'))

    M = calculate_projection_matrix(Points_2D, Points_3D)
    print(f'The projection matrix:\n{M}\n')

    Projected_2D_Pts, Residual = evaluate_points(M, Points_2D, Points_3D)
    print(f'Total residual:\n{Residual}\n')

    if not args.no_vis:
        visualize_points(Points_2D, Projected_2D_Pts)

    Center = compute_camera_center(M)
    print(f'Estimated camera location:\n{Center}\n')

    if not args.no_vis:
        plot3dview(Points_3D, Center)

    Points_2D_pic_a = np.loadtxt(os.path.join(data_dir, 'pts2d-pic_a.txt'))
    Points_2D_pic_b = np.loadtxt(os.path.join(data_dir, 'pts2d-pic_b.txt'))

    ImgLeft = io.imread(os.path.join(data_dir, 'pic_a.jpg'))
    ImgRight = io.imread(os.path.join(data_dir, 'pic_b.jpg'))

    F_matrix = estimate_fundamental_matrix(Points_2D_pic_a, Points_2D_pic_b)

    if not args.no_vis:
        draw_epipolar_lines(F_matrix, ImgLeft, ImgRight, Points_2D_pic_a, Points_2D_pic_b)

    print(f"Using image: {args.image}")

    if args.image == "mt_rushmore":
        pic_a = io.imread(os.path.join(data_dir, 'MountRushmore/Mount_Rushmore1.jpg'))
        pic_b = io.imread(os.path.join(data_dir, 'MountRushmore/Mount_Rushmore2.jpg'))
        sf = pic_b.shape[0] / pic_a.shape[0]
        Points_2D_pic_a, Points_2D_pic_b = get_ground_truth(
            os.path.join(data_dir, "MountRushmore/mt_rushmore.mat"), scale_factor_A=sf
        )
        pic_a = transform.rescale(pic_a, sf, channel_axis=2)

    elif args.image == "notre_dame":
        pic_a = io.imread(os.path.join(data_dir, 'NotreDame/NotreDame1.jpg'))
        pic_b = io.imread(os.path.join(data_dir, 'NotreDame/NotreDame2.jpg'))
        sf = pic_b.shape[0] / pic_a.shape[0]
        Points_2D_pic_a, Points_2D_pic_b = get_ground_truth(
            os.path.join(data_dir, "NotreDame/notre_dame.mat"), scale_factor_A=sf
        )
        pic_a = transform.rescale(pic_a, sf, channel_axis=2)

    elif args.image == "gaudi":
        pic_a = io.imread(os.path.join(data_dir, 'EpiscopalGaudi/EGaudi_1.jpg'))
        pic_b = io.imread(os.path.join(data_dir, 'EpiscopalGaudi/EGaudi_2.jpg'))
        sf = pic_b.shape[0] / pic_a.shape[0]
        Points_2D_pic_a, Points_2D_pic_b = get_ground_truth(
            os.path.join(data_dir, "EpiscopalGaudi/gaudi.mat"), scale_factor_A=sf
        )
        pic_a = transform.rescale(pic_a, sf, channel_axis=2)

    else:
        print(f"Error: Invalid image argument '{args.image}'")
        return

    if args.positional_ratio:
        print('Applying noise on position')
        Points_2D_pic_a = apply_positional_noise(Points_2D_pic_a, pic_a.shape[0] * sf, pic_a.shape[1] * sf, args.positional_interval, args.positional_ratio)
        Points_2D_pic_b = apply_positional_noise(Points_2D_pic_b, pic_b.shape[0], pic_b.shape[1], args.positional_interval, args.positional_ratio)

    if args.use_orb:
        print('Using ORB')
        Points_2D_pic_a, Points_2D_pic_b = matchAndShowCorrespondence(pic_a, pic_b)
        print(f'Found {Points_2D_pic_a.shape[0]} possibly matching features using ORB\n')

    if args.matching_ratio:
        print('Applying noise on matches')
        Points_2D_pic_a = apply_matching_noise(Points_2D_pic_a, args.matching_ratio)

    if args.no_ransac:
        print('Skipping RANSAC; estimating fundamental matrix directly.')
        F_matrix = estimate_fundamental_matrix(Points_2D_pic_a, Points_2D_pic_b)
        matched_points_a, matched_points_b = Points_2D_pic_a, Points_2D_pic_b
    else:
        print('Running RANSAC for fundamental matrix estimation.')
        F_matrix, matched_points_a, matched_points_b = ransac_fundamental_matrix(Points_2D_pic_a, Points_2D_pic_b)

    if not args.no_vis:
        H, _ = cv2.findHomography(matched_points_a, matched_points_b)
        pic_a = cv2.warpPerspective(pic_a, H, (pic_b.shape[1], pic_b.shape[0]))
        transformed_points_a = cv2.perspectiveTransform(matched_points_a.reshape(-1, 1, 2), H).squeeze(axis=1)
        showCorrespondence(pic_a, pic_b, transformed_points_a, matched_points_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, choices=['mt_rushmore', 'notre_dame', 'gaudi'], default='mt_rushmore', help="Specify dataset")
    parser.add_argument("--positional-interval", type=int, default=3, help="Range for positional noise")
    parser.add_argument("--positional-ratio", type=float, default=0.2, help="Ratio of points to perturb")
    parser.add_argument("--matching-ratio", type=float, default=0.2, help="Ratio of matches to shuffle")
    parser.add_argument("--hard_points", action="store_true", help="Use harder, unnormalized points")
    parser.add_argument("--no-ransac", action="store_true", help="Disable RANSAC for fundamental matrix estimation")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--use-orb", action="store_true", help="Use ORB for feature matching")

    args = parser.parse_args()
    main(args)

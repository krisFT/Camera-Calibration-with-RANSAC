# Camera Calibration with RANSAC

## Overview
This project implements camera calibration using projection matrix estimation, camera center computation, and fundamental matrix estimation. It also applies RANSAC, implemented from scratch, to robustly filter outliers, ensuring accurate epipolar geometry estimation for image correspondence visualization.

## Features
- Projection Matrix Calculation
- Camera Center Estimation
- Fundamental Matrix Estimation
- RANSAC-based Outlier Removal
- Epipolar Line Visualization

## Explanation of Key Implementations

### Projection Matrix Calculation
- The projection matrix is computed by solving a system of linear equations that maps 3D world points to 2D image points. This is done by constructing a matrix A based on the known correspondences and solving for the projection matrix M using least squares.

### Camera Center Computation
- The camera center is extracted from the projection matrix by computing the null space of the first three columns of M. This is achieved using pseudo-inverse decomposition, ensuring an accurate estimation of the camera’s world coordinates.

### Fundamental Matrix Estimation
- The fundamental matrix encodes the epipolar geometry between two images. It is estimated using a set of corresponding points and is refined using Singular Value Decomposition (SVD) to enforce the rank-2 constraint. The matrix is further improved using RANSAC to eliminate outliers.

## Installation & Setup
### Clone the Repository
```sh
git clone https://github.com/krisFT/camera-calibration-ransac.git
cd camera-calibration-ransac
```

## Usage
Run the main script to perform camera calibration and visualize results:
```sh
python main.py --image mt_rushmore
```

### Command-Line Arguments
| Argument | Description |
|----------|-------------|
| `--image` | Choose image set: `mt_rushmore`, `notre_dame`, `gaudi` |
| `--hard_points` | Use unnormalized points |
| `--no-vis` | Disable visualization |
| `--no-ransac` | Disable RANSAC |
| `--use-orb` | Use ORB feature matching instead of ground truth |


## Acknowledgments
This project is based on work from CSCI 1430 @ Brown and CS 4495/6476 @ Georgia Tech.



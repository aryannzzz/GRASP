import numpy as np
import cv2 as cv
import json

def load_calibration(json_path):
    """Loads the unique camera matrix and distortion coefficients from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    return camera_matrix, dist_coeffs

def pixel_to_camera_coords(u, v, depth_z, camera_matrix, dist_coeffs):
    """
    Converts a 2D pixel into 3D coordinates relative to the CAMERA LENS (0,0,0).
    """
    # Format pixel for OpenCV
    pixel_array = np.array([[[float(u), float(v)]]], dtype=np.float32)

    # Mathematically remove lens distortion using the unique coefficients
    normalized_point = cv.undistortPoints(pixel_array, camera_matrix, dist_coeffs)

    # Scale the normalized ray by the physical depth of the object
    X_cam = normalized_point[0][0][0] * depth_z
    Y_cam = normalized_point[0][0][1] * depth_z
    Z_cam = depth_z

    return np.array([X_cam, Y_cam, Z_cam])
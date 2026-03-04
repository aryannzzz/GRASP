import numpy as np
import cv2 as cv
import glob
import sys
import json

# 1. Setup checkerboard dimensions (Inner corners)
CHECKERBOARD_SIZE = (6, 9) 

# Termination criteria for sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points: (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = [] # 3D points in real-world space
imgpoints = [] # 2D points in the image plane

# Load images (supports both .jpg and .jpeg)
images = glob.glob('*.jpg') + glob.glob('*.jpeg')

if len(images) == 0:
    print("Error: No '.jpg' or '.jpeg' images found in the current directory.")
    sys.exit()

print(f"Found {len(images)} images. Processing...")

# 2. Extract corners from all images
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Optional visual feedback
        cv.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        cv.namedWindow('Detected Corners', cv.WINDOW_NORMAL) 
        cv.imshow('Detected Corners', img)
        cv.waitKey(100) # Fast playback

cv.destroyAllWindows()

# 3. Perform Camera Calibration
if len(objpoints) > 0:
    print("\nCalibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n--- Calibration Successful ---")
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    # Calculate Re-projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\nTotal Re-projection Error: {mean_error/len(objpoints):.4f}")

    # 4. Save the Calibration Data to JSON
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist()
    }
    
    with open('camera_calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)
        
    print("\nSUCCESS: Saved matrix and coefficients to 'camera_calibration.json'")

    # 5. Undistort the first image as a visual test
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.jpg', dst)
    print("Saved undistorted test image as 'calibresult.jpg'")

else:
    print("\nError: Could not find the checkerboard pattern in any images.")
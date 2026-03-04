import numpy as np
from pixel_to_world import load_calibration, pixel_to_camera_coords

# --- 1. PHYSICAL SETUP MEASUREMENTS (Configure these once!) ---
# Measure the physical distance from the Robot's Base (0,0,0) to the Camera Lens.
# Example: The camera is mounted 400mm in front of the arm, and 500mm above the table.
OFFSET_X = 400.0  
OFFSET_Y = 0.0    
OFFSET_Z = 500.0  

def camera_to_robot_base(cam_xyz):
    """
    Shifts the 3D origin from the Camera Lens down to the Robot Base.
    Assumes a fixed, top-down camera setup looking at a table.
    """
    X_cam, Y_cam, Z_cam = cam_xyz
    
    # Apply Translation (Offset) and basic Rotation (flipping axes)
    # Note: You may need to invert X or Y (+/-) depending on which way 
    # your specific robot arm defines "forward" and "right".
    X_robot = OFFSET_X + X_cam 
    Y_robot = OFFSET_Y - Y_cam  
    Z_robot = OFFSET_Z - Z_cam  

    return np.array([X_robot, Y_robot, Z_robot])


if __name__ == "__main__":
    # --- 2. INITIALIZATION ---
    # Load calibration data once when the system boots
    cam_matrix, dist_coeffs = load_calibration('camera_calibration.json')
    print("System Ready: Camera parameters loaded.\n")

    # --- 3. LIVE VISUOMOTOR LOOP ---
    # In reality, these come directly from your YOLO/vision-language model output
    target_pixel_u = 1024  
    target_pixel_v = 768   
    
    # The physical distance from the lens to the table/object (measured in mm)
    known_depth_mm = 500.0  

    # Step A: Get 3D coordinates relative to the lens
    camera_xyz = pixel_to_camera_coords(
        u=target_pixel_u, 
        v=target_pixel_v, 
        depth_z=known_depth_mm, 
        camera_matrix=cam_matrix, 
        dist_coeffs=dist_coeffs
    )

    # Step B: Transform origin to the robot base
    robot_xyz = camera_to_robot_base(camera_xyz)

    # --- 4. OUTPUT TO HARDWARE ---
    print(f"Target located at Pixel: ({target_pixel_u}, {target_pixel_v})")
    print(f"\n[Camera Frame] (Lens is 0,0,0):")
    print(f"X: {camera_xyz[0]:.2f} mm | Y: {camera_xyz[1]:.2f} mm | Z: {camera_xyz[2]:.2f} mm")
    
    print(f"\n[Robot Base Frame] (Arm base is 0,0,0) -> SEND TO KINEMATICS:")
    print(f"X: {robot_xyz[0]:.2f} mm | Y: {robot_xyz[1]:.2f} mm | Z: {robot_xyz[2]:.2f} mm")
"""
Generate printable ArUco markers for object detection.

Run this script to create markers that you can print and attach to objects.
"""

import cv2
import numpy as np
from pathlib import Path


def generate_aruco_markers(
    output_dir: str = "./aruco_markers",
    num_markers: int = 6,
    marker_size_px: int = 400,
    dictionary: int = cv2.aruco.DICT_4X4_50,
    border_bits: int = 1
):
    """
    Generate ArUco markers as PNG images.
    
    Args:
        output_dir: Where to save marker images
        num_markers: How many markers to generate
        marker_size_px: Size of each marker in pixels
        dictionary: ArUco dictionary to use
        border_bits: Border size (1 = thinner, 2 = thicker)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    
    print(f"\n{'='*60}")
    print(f"GENERATING ARUCO MARKERS")
    print(f"{'='*60}\n")
    
    # Object name mapping
    marker_names = {
        0: "tape",
        1: "target_zone",
        2: "table",
        3: "obstacle",
        4: "container",
        5: "tool"
    }
    
    # Generate markers
    for marker_id in range(num_markers):
        # Generate marker image
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            marker_size_px,
            borderBits=border_bits
        )
        
        # Add white border for printing
        border_px = 50
        bordered_image = np.ones(
            (marker_size_px + 2*border_px, marker_size_px + 2*border_px),
            dtype=np.uint8
        ) * 255
        bordered_image[border_px:-border_px, border_px:-border_px] = marker_image
        
        # Add text label
        object_name = marker_names.get(marker_id, f"marker_{marker_id}")
        text = f"ID: {marker_id} - {object_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add text at bottom
        text_x = (bordered_image.shape[1] - text_width) // 2
        text_y = bordered_image.shape[0] - 20
        
        cv2.putText(
            bordered_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            0,
            thickness
        )
        
        # Save marker
        filename = output_path / f"marker_{marker_id}_{object_name}.png"
        cv2.imwrite(str(filename), bordered_image)
        
        print(f"✓ Generated: {filename}")
        print(f"  ID: {marker_id}")
        print(f"  Object: {object_name}")
        print()
    
    # Generate sheet with all markers
    print("Generating combined sheet...")
    generate_marker_sheet(
        output_path,
        num_markers,
        aruco_dict,
        marker_names,
        marker_size_px // 2
    )
    
    # Print instructions
    print(f"\n{'='*60}")
    print("PRINTING INSTRUCTIONS")
    print(f"{'='*60}\n")
    
    print("1. Print the generated PNG files")
    print("   - Use white paper")
    print("   - Print at actual size (100% scale)")
    print("   - Use high quality / best quality setting")
    print()
    
    print("2. Cut out the markers")
    print("   - Keep the white border")
    print("   - Cut as straight as possible")
    print()
    
    print("3. Attach to objects")
    print("   - Marker 0 → tape")
    print("   - Marker 1 → target zone")
    print("   - Marker 2 → table")
    print("   - (Add more as needed)")
    print()
    
    print("4. Marker size for detection")
    print("   - Measure the BLACK square size (not border)")
    print("   - Use this value for --marker-size parameter")
    print("   - Example: 5cm marker → --marker-size 0.05")
    print()
    
    print("5. Test detection")
    print("   python perception_system.py")
    print()


def generate_marker_sheet(
    output_path: Path,
    num_markers: int,
    aruco_dict,
    marker_names: dict,
    marker_size_px: int = 200
):
    """Generate a single sheet with all markers."""
    
    # Create sheet (A4 proportions: 210mm x 297mm ≈ 2:3)
    sheet_width = 2100
    sheet_height = 2970
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    # Calculate grid
    markers_per_row = 3
    spacing_x = sheet_width // markers_per_row
    spacing_y = 400
    
    # Add title
    title = "ArUco Markers for RoboPrompt"
    cv2.putText(
        sheet, title, (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4
    )
    
    # Place markers in grid
    y_offset = 200
    
    for i in range(num_markers):
        row = i // markers_per_row
        col = i % markers_per_row
        
        # Generate marker
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, i, marker_size_px
        )
        
        # Calculate position (centered in cell)
        x = col * spacing_x + (spacing_x - marker_size_px) // 2
        y = y_offset + row * spacing_y
        
        # Place marker
        if y + marker_size_px < sheet_height:
            sheet[y:y+marker_size_px, x:x+marker_size_px] = marker_img
            
            # Add label
            object_name = marker_names.get(i, f"marker_{i}")
            label = f"ID {i}: {object_name}"
            
            # Center label under marker
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (marker_size_px - text_size[0]) // 2
            text_y = y + marker_size_px + 30
            
            cv2.putText(
                sheet, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2
            )
    
    # Add footer
    footer = "Print at 100% scale on white paper"
    cv2.putText(
        sheet, footer, (50, sheet_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2
    )
    
    # Save sheet
    sheet_path = output_path / "all_markers_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"✓ Generated combined sheet: {sheet_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ArUco markers for RoboPrompt')
    parser.add_argument('--output', type=str, default='./aruco_markers',
                       help='Output directory')
    parser.add_argument('--num', type=int, default=6,
                       help='Number of markers to generate')
    parser.add_argument('--size', type=int, default=400,
                       help='Marker size in pixels')
    
    args = parser.parse_args()
    
    generate_aruco_markers(
        output_dir=args.output,
        num_markers=args.num,
        marker_size_px=args.size
    )
    
    print("\n✅ DONE!")
    print(f"\nMarkers saved to: {args.output}/")
    print("\nNext steps:")
    print("1. Print the PNG files")
    print("2. Cut out and attach to objects")
    print("3. Test with: python perception_system.py")


if __name__ == "__main__":
    main()

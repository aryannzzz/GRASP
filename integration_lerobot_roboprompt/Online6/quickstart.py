#!/usr/bin/env python3
"""
QuickStart Script for RoboPrompt + SO-101 Integration

This script guides you through the complete workflow step-by-step.
"""

import sys
import subprocess
from pathlib import Path
import json


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def print_step(step_num, text):
    """Print a step number and description."""
    print(f"\n{'🔹' if step_num else '⚡'} {text}")


def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()


def run_command(cmd, description):
    """Run a command and show output."""
    print(f"\n▶️  {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: Command not found. Make sure dependencies are installed.")
        return False


def main():
    """Main quickstart workflow."""
    
    print_header("ROBOPROMPT + SO-101 QUICKSTART")
    
    print("This script will guide you through:")
    print("  1. Setup and installation")
    print("  2. Generating ArUco markers")
    print("  3. Creating ICL demonstrations")
    print("  4. Testing perception")
    print("  5. Running inference")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check dependencies
    print_header("STEP 1: Check Dependencies")
    
    print("Checking if key dependencies are installed...")
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - run: pip install numpy")
    
    try:
        import cv2
        print("✅ opencv")
    except ImportError:
        print("❌ opencv - run: pip install opencv-contrib-python")
    
    try:
        import lerobot
        print("✅ lerobot")
    except ImportError:
        print("❌ lerobot - run: pip install lerobot")
    
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
    
    cont = input("\nDependencies installed? (y/n): ")
    if cont.lower() != 'y':
        print("Please install dependencies first, then run this script again.")
        return
    
    # Step 2: Generate ArUco markers
    print_header("STEP 2: Generate ArUco Markers")
    
    if check_file_exists("./aruco_markers/marker_0_tape.png"):
        print("✓ ArUco markers already generated")
        regen = input("Regenerate markers? (y/n): ")
        if regen.lower() == 'y':
            run_command(
                ["python", "generate_aruco_markers.py"],
                "Generating ArUco markers"
            )
    else:
        print("Generating ArUco markers for the first time...")
        run_command(
            ["python", "generate_aruco_markers.py"],
            "Generating ArUco markers"
        )
    
    print("\n📋 Next: Print the markers in ./aruco_markers/ directory")
    print("   - Print at 100% scale")
    print("   - Cut out and attach to objects:")
    print("     • Marker 0 → tape")
    print("     • Marker 1 → target zone")
    print("     • Marker 2 → table")
    
    input("\nPress Enter when markers are printed and attached...")
    
    # Step 3: Create ICL demonstrations
    print_header("STEP 3: Create ICL Demonstrations")
    
    if check_file_exists("./roboprompt_data/icl_demos.json"):
        print("✓ ICL demonstrations already created")
        recreate = input("Recreate demonstrations? (y/n): ")
        if recreate.lower() != 'y':
            print("Skipping demo creation...")
        else:
            run_command(
                ["python", "lerobot_roboprompt_dataset.py"],
                "Creating ICL demonstrations from dataset"
            )
    else:
        print("Creating ICL demonstrations from LeRobot dataset...")
        print("This will download the dataset from HuggingFace (may take a few minutes)")
        
        run_command(
            ["python", "lerobot_roboprompt_dataset.py"],
            "Creating ICL demonstrations"
        )
    
    # Verify demos were created
    if not check_file_exists("./roboprompt_data/icl_demos.json"):
        print("\n❌ Failed to create ICL demonstrations")
        print("   Check error messages above and try again")
        return
    
    print("\n✅ ICL demonstrations created successfully!")
    
    # Show demo info
    try:
        with open("./roboprompt_data/icl_demos.json") as f:
            icl_data = json.load(f)
        
        print(f"\n📊 Demo Information:")
        print(f"   Task: {icl_data['task']}")
        print(f"   Episodes used: {len(icl_data['episodes_used'])}")
        print(f"   Total keyframes: {sum(icl_data['num_keyframes'])}")
    except Exception as e:
        print(f"⚠️  Could not read demo file: {e}")
    
    # Step 4: Test perception
    print_header("STEP 4: Test Perception System")
    
    print("Testing object detection with ArUco markers...")
    print("\n⚡ Place objects with markers in camera view")
    print("   Press 'q' to quit perception test")
    
    test_perception = input("\nTest perception now? (y/n): ")
    if test_perception.lower() == 'y':
        run_command(
            ["python", "perception_system.py"],
            "Testing perception system"
        )
    
    # Step 5: Run inference
    print_header("STEP 5: Run Inference Pipeline")
    
    print("Choose how to run inference:")
    print("  1. Simulation mode (no robot movement)")
    print("  2. Real robot (slow speed)")
    print("  3. Real robot (custom speed)")
    print("  4. Skip for now")
    
    choice = input("\nYour choice (1-4): ")
    
    if choice == '1':
        print("\n🎮 Running in SIMULATION mode")
        run_command(
            ["python", "roboprompt_so101_pipeline.py", "--sim"],
            "Running inference (simulation)"
        )
    
    elif choice == '2':
        robot_port = input("Enter robot port (default: /dev/ttyUSB0): ") or "/dev/ttyUSB0"
        print(f"\n🤖 Running on REAL ROBOT (port: {robot_port})")
        print("⚠️  Safety reminder:")
        print("   - Robot will move slowly (speed: 0.3)")
        print("   - Keep e-stop accessible")
        print("   - Clear workspace of obstacles")
        
        confirm = input("\nReady to proceed? (y/n): ")
        if confirm.lower() == 'y':
            run_command(
                ["python", "roboprompt_so101_pipeline.py", 
                 "--speed", "0.3", "--robot-port", robot_port],
                "Running inference on real robot"
            )
    
    elif choice == '3':
        robot_port = input("Enter robot port (default: /dev/ttyUSB0): ") or "/dev/ttyUSB0"
        speed = input("Enter speed (0.1-2.0, default: 0.5): ") or "0.5"
        
        print(f"\n🤖 Running on REAL ROBOT (port: {robot_port}, speed: {speed})")
        print("⚠️  Safety reminder:")
        print("   - Keep e-stop accessible")
        print("   - Clear workspace of obstacles")
        print("   - Monitor robot closely")
        
        confirm = input("\nReady to proceed? (y/n): ")
        if confirm.lower() == 'y':
            run_command(
                ["python", "roboprompt_so101_pipeline.py",
                 "--speed", speed, "--robot-port", robot_port],
                "Running inference on real robot"
            )
    
    else:
        print("\nSkipping inference for now.")
        print("\nTo run inference later:")
        print("  python roboprompt_so101_pipeline.py --sim  # simulation")
        print("  python roboprompt_so101_pipeline.py --speed 0.3  # real robot")
    
    # Summary
    print_header("QUICKSTART COMPLETE!")
    
    print("✅ What we did:")
    print("  ✓ Checked dependencies")
    print("  ✓ Generated ArUco markers")
    print("  ✓ Created ICL demonstrations")
    print("  ✓ Tested perception (optional)")
    print("  ✓ Ran inference (optional)")
    
    print("\n📚 Next Steps:")
    print("  1. Review COMPLETE_GUIDE.md for detailed documentation")
    print("  2. Integrate real LLM (currently using placeholder)")
    print("  3. Collect more demonstrations")
    print("  4. Try different tasks")
    
    print("\n🔧 Useful Commands:")
    print("  # Test perception anytime")
    print("  python perception_system.py")
    print()
    print("  # Run inference in simulation")
    print("  python roboprompt_so101_pipeline.py --sim")
    print()
    print("  # Run inference on real robot")
    print("  python roboprompt_so101_pipeline.py --speed 0.3")
    print()
    print("  # Generate new markers")
    print("  python generate_aruco_markers.py")
    
    print("\n🎉 Happy robot programming!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)

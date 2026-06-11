import os
import sys
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from datetime import datetime
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# file selection menu
DATA_DIR = "data"
OUTPUT_DIR = "output"

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' could not be found.")
    sys.exit(1)

# Scan for valid TIFF files
valid_extensions = ('.tif', '.tiff')
tiff_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(valid_extensions)]

if not tiff_files:
    print(f"Error: No matching tracking files (.tif or .tiff) found in '{DATA_DIR}'.")
    sys.exit(1)

# Render the Text User Interface
print("\n" + "=" * 60)
print("             PARTICLE TRACKER MENU             ")
print("=" * 60)
for idx, filename in enumerate(tiff_files, 1):
    print(f"  [{idx}] {filename}")
print("=" * 60)

# Loop until the user provides valid input
while True:
    try:
        user_input = input("Select a file number to process (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Operation cancelled. Exiting program.")
            sys.exit(0)

        selection_idx = int(user_input) - 1
        if 0 <= selection_idx < len(tiff_files):
            TIFF_FILENAME = tiff_files[selection_idx]
            break
        else:
            print(f"Out of bounds. Choose a number between 1 and {len(tiff_files)}.")
    except ValueError:
        print("Invalid entry. Please enter a valid menu integer.")

# set up run directories
tiff_path = os.path.join(DATA_DIR, TIFF_FILENAME)
video_stack = tiff.imread(tiff_path)
total_frames = len(video_stack)

print(f"\n[+] Successfully loaded '{TIFF_FILENAME}' ({total_frames} frames).")

# Create a unique timestamped session folder based on execution time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Include the base filename in the folder title for easier reference later
file_slug = os.path.splitext(TIFF_FILENAME)[0]
run_dir = os.path.join(OUTPUT_DIR, f"run_{file_slug}_{timestamp}")
frames_dir = os.path.join(run_dir, "annotated_frames")

os.makedirs(frames_dir, exist_ok=True)
print(f"[+] Initialized output directory: {run_dir}")

# tracking parameters
Y_MIN, Y_MAX = 48, 180
X_MIN, X_MAX = 0, 511

tracking_data = []

print("[+] Processing frames and generating annotations...")

#core processing loop
for frame_idx in range(total_frames):
    raw_frame = video_stack[frame_idx]

    # crop to region of Interest
    cropped_frame = raw_frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

    # thresholding
    thresh_value = threshold_otsu(cropped_frame)
    binary_frame = cropped_frame < thresh_value

    # remove the top bar artifacts to isolate the channel
    binary_frame[0:15, :] = False

    # Group and label white pixels
    label_image = label(binary_frame)
    regions = regionprops(label_image)

    target_particle = None

    for prop in regions:
        if prop.area < 50:  # filter out pixel noise
            continue
        target_particle = prop
        break

    # Prepare background image for color annotations (Convert to 8-bit BGR color space)
    img_normalized = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    annotated_frame = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

    if target_particle is not None:
        # Extract location and orientation
        cy, cx = target_particle.centroid
        orientation = target_particle.orientation

        # Calculate changes relative to the previous frame
        if len(tracking_data) == 0:
            dx, dy = 0.0, 0.0
            d_theta = 0.0
        else:
            dx = cx - tracking_data[-1]['Centroid_X']
            dy = cy - tracking_data[-1]['Centroid_Y']

            # Rotational Delta with angle wrapping fix
            prev_orientation = tracking_data[-1]['Orientation']
            d_theta = orientation - prev_orientation
            if d_theta > np.pi / 2:
                d_theta -= np.pi
            elif d_theta < -np.pi / 2:
                d_theta += np.pi

        # Update data registry
        tracking_data.append({
            'Frame': frame_idx,
            'Centroid_X': cx,
            'Centroid_Y': cy,
            'Displacement_X': dx,
            'Displacement_Y': dy,
            'Orientation': orientation,
            'Rotation_Delta': d_theta
        })

        # add visual annotations
        # 1. Green Bounding Box
        min_row, min_col, max_row, max_col = target_particle.bbox
        cv2.rectangle(annotated_frame, (min_col, min_row), (max_col, max_row), (0, 255, 0), 1)

        # 2. Red Centroid Dot
        cv2.circle(annotated_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # 3. Dynamic Telemetry Text (Top Left Corner)
        text_coords = f"Centroid: ({cx:.1f}, {cy:.1f})"
        text_rot = f"d_Theta: {d_theta:.3f} rad"

        cv2.putText(annotated_frame, text_coords, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, text_rot, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    else:
        # Log a warning on the image if the cluster is lost/obscured
        cv2.putText(annotated_frame, "PARTICLE LOST", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Save the annotated frame to disk
    frame_output_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_output_path, annotated_frame)

# export data to csv
df = pd.DataFrame(tracking_data)
csv_output_path = os.path.join(run_dir, "particle_tracks.csv")
df.to_csv(csv_output_path, index=False)

print("\nProcessing Complete!")
print(f"-> Data Table: {csv_output_path}")
print(f"-> Frame Visuals Saved: {total_frames} images inside {frames_dir}\n")

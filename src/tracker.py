import os
import sys
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from datetime import datetime
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# ==========================================
# 1. TUI COMPONENT - FILE & OUTPUT SELECTION
# ==========================================
DATA_DIR = "../data"
OUTPUT_DIR = "../output"

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' could not be found.")
    sys.exit(1)

valid_extensions = ('.tif', '.tiff')
tiff_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(valid_extensions)]

if not tiff_files:
    print(f"Error: No matching tracking files (.tif or .tiff) found in '{DATA_DIR}'.")
    sys.exit(1)

print("\n" + "=" * 60)
print("         MULTI-OBJECT MICROSCOPY TRACKER SYSTEM           ")
print("=" * 60)
for idx, filename in enumerate(tiff_files, 1):
    print(f"  [{idx}] {filename}")
print("=" * 60)

# Menu A: Source File Selection
while True:
    try:
        user_input = input("Select a file number to process (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Exiting program.")
            sys.exit(0)

        selection_idx = int(user_input) - 1
        if 0 <= selection_idx < len(tiff_files):
            TIFF_FILENAME = tiff_files[selection_idx]
            break
        else:
            print(f"Out of bounds. Choose a number between 1 and {len(tiff_files)}.")
    except ValueError:
        print("Invalid entry. Please enter an integer.")

# Menu B: Export Format Selection Toggle
print("\n" + "-" * 40)
print("  SELECT OUTPUT EXPORT FORMAT")
print("----------------------------------------")
print("  [1] Individual PNG Frames Folder (High-disk space / Scannable)")
print("  [2] Compiled MP4 Video File     (Low-disk space / Shareable)")
print("-" * 40)

while True:
    output_selection = input("Select export mode [1 or 2]: ").strip()
    if output_selection in ('1', '2'):
        EXPORT_MODE = "frames" if output_selection == '1' else "video"
        break
    print("Invalid option. Please enter 1 for Frames or 2 for Video.")

# ==========================================
# 2. SETUP RUN DIRECTORIES
# ==========================================
tiff_path = os.path.join(DATA_DIR, TIFF_FILENAME)
video_stack = tiff.imread(tiff_path)
total_frames = len(video_stack)

print(f"\n[+] Loaded '{TIFF_FILENAME}' ({total_frames} frames).")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_slug = os.path.splitext(TIFF_FILENAME)[0]
run_dir = os.path.join(OUTPUT_DIR, f"run_{file_slug}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

frames_dir = None
video_writer = None

if EXPORT_MODE == "frames":
    frames_dir = os.path.join(run_dir, "annotated_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[+] Output Target: Frames directory -> {frames_dir}")
else:
    print(f"[+] Output Target: Compressed MP4 Video File Container")

# ==========================================
# 3. TRACKING & CLASSIFICATION PARAMETERS
# ==========================================
Y_MIN, Y_MAX = 48, 180
X_MIN, X_MAX = 0, 511

MAX_TRACKING_DISTANCE = 60
EDGE_BUFFER_ZONE = 35

cluster_counter = 1
single_counter = 1
internal_id_counter = 1

active_objects = {}
all_frames_data = []

# ==========================================
# 4. CORE MULTI-OBJECT LOOP
# ==========================================
print(f"[+] Processing timeline via ({EXPORT_MODE.upper()}) pipeline...")

for frame_idx in range(total_frames):
    raw_frame = video_stack[frame_idx]
    cropped_frame = raw_frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

    # Preprocessing & Clean masking
    thresh_value = threshold_otsu(cropped_frame)
    binary_frame = cropped_frame < thresh_value
    binary_frame[0:15, :] = False

    # Feature extraction
    label_image = label(binary_frame)
    regions = regionprops(label_image)

    current_detections = []
    for prop in regions:
        if prop.area < 40:
            continue

        cy, cx = prop.centroid

        if prop.area > 350 or prop.eccentricity > 0.60:
            instant_class = "cluster"
        else:
            instant_class = "single"

        current_detections.append({
            'centroid': (cx, cy),
            'orientation': prop.orientation,
            'bbox': prop.bbox,
            'instant_class': instant_class
        })

    # Standardize canvas
    img_normalized = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    annotated_frame = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

    matched_current_indices = set()
    updated_active_objects = {}

    # --- DATA ASSOCIATION (CENTROID MATCHING) ---
    if active_objects and current_detections:
        pairs = []
        for int_id, obj_data in active_objects.items():
            for det_idx, det_data in enumerate(current_detections):
                dist = np.linalg.norm(np.array(obj_data['centroid']) - np.array(det_data['centroid']))
                pairs.append((dist, int_id, det_idx))

        pairs.sort(key=lambda x: x[0])

        assigned_internal_ids = set()
        for dist, int_id, det_idx in pairs:
            if int_id in assigned_internal_ids or det_idx in matched_current_indices:
                continue

            if dist <= MAX_TRACKING_DISTANCE:
                det_data = current_detections[det_idx]
                matched_current_indices.add(det_idx)
                assigned_internal_ids.add(int_id)

                obj_data = active_objects[int_id]
                cx, cy = det_data['centroid']

                dx = cx - obj_data['centroid'][0]
                dy = cy - obj_data['centroid'][1]
                d_theta = det_data['orientation'] - obj_data['orientation']
                if d_theta > np.pi / 2: d_theta -= np.pi
                elif d_theta < -np.pi / 2: d_theta += np.pi

                # --- ONE-WAY UPGRADE & PROBATION LOGIC ---
                if det_data['instant_class'] == "cluster":
                    if obj_data['final_id'] is None and cx >= EDGE_BUFFER_ZONE:
                        obj_data['final_id'] = f"cluster{cluster_counter:02d}"
                        obj_data['assigned_class'] = "cluster"
                        cluster_counter += 1
                        for h_idx in obj_data['database_indices']:
                            all_frames_data[h_idx]['Object_ID'] = obj_data['final_id']
                            all_frames_data[h_idx]['Class'] = obj_data['assigned_class']

                    elif obj_data['assigned_class'] == "single":
                        obj_data['final_id'] = f"cluster{cluster_counter:02d}"
                        obj_data['assigned_class'] = "cluster"
                        cluster_counter += 1
                        for h_idx in obj_data['database_indices']:
                            all_frames_data[h_idx]['Object_ID'] = obj_data['final_id']
                            all_frames_data[h_idx]['Class'] = obj_data['assigned_class']

                elif det_data['instant_class'] == "single":
                    if obj_data['final_id'] is None and cx >= EDGE_BUFFER_ZONE:
                        obj_data['final_id'] = f"single{single_counter:02d}"
                        obj_data['assigned_class'] = "single"
                        single_counter += 1
                        for h_idx in obj_data['database_indices']:
                            all_frames_data[h_idx]['Object_ID'] = obj_data['final_id']
                            all_frames_data[h_idx]['Class'] = obj_data['assigned_class']

                updated_active_objects[int_id] = {
                    'centroid': det_data['centroid'],
                    'orientation': det_data['orientation'],
                    'final_id': obj_data['final_id'],
                    'assigned_class': obj_data['assigned_class'],
                    'database_indices': obj_data['database_indices']
                }

                db_row_idx = len(all_frames_data)
                obj_data['database_indices'].append(db_row_idx)

                display_id = obj_data['final_id'] if obj_data['final_id'] else f"pending_{int_id}"
                display_class = obj_data['assigned_class'] if obj_data['assigned_class'] else "pending"

                all_frames_data.append({
                    'Frame': frame_idx,
                    'Object_ID': display_id,
                    'Class': display_class,
                    'Centroid_X': cx,
                    'Centroid_Y': cy,
                    'Displacement_X': dx,
                    'Displacement_Y': dy,
                    'Orientation': det_data['orientation'],
                    'Rotation_Delta': d_theta
                })

                min_row, min_col, max_row, max_col = det_data['bbox']
                box_color = (0, 255, 0) if display_class == "single" else (255, 0, 0)
                if display_class == "pending": box_color = (0, 165, 255)

                cv2.rectangle(annotated_frame, (min_col, min_row), (max_col, max_row), box_color, 1)
                cv2.circle(annotated_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                text_y = max(15, min_row - 8)
                cv2.putText(annotated_frame, f"{display_id}", (min_col, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1, cv2.LINE_AA)
                if obj_data['final_id']:
                    cv2.putText(annotated_frame, f"rot: {d_theta:.2f}", (min_col, text_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1, cv2.LINE_AA)

    # --- REGISTRATION FOR NEW ENTITIES ---
    for det_idx, det_data in enumerate(current_detections):
        if det_idx in matched_current_indices:
            continue

        cx, cy = det_data['centroid']
        final_id = None
        assigned_class = None

        if cx >= EDGE_BUFFER_ZONE:
            if det_data['instant_class'] == "cluster":
                final_id = f"cluster{cluster_counter:02d}"
                assigned_class = "cluster"
                cluster_counter += 1
            else:
                final_id = f"single{single_counter:02d}"
                assigned_class = "single"
                single_counter += 1

        int_id = internal_id_counter
        internal_id_counter += 1

        db_row_idx = len(all_frames_data)
        display_id = final_id if final_id else f"pending_{int_id}"
        display_class = assigned_class if assigned_class else "pending"

        updated_active_objects[int_id] = {
            'centroid': det_data['centroid'],
            'orientation': det_data['orientation'],
            'final_id': final_id,
            'assigned_class': assigned_class,
            'database_indices': [db_row_idx]
        }

        all_frames_data.append({
            'Frame': frame_idx,
            'Object_ID': display_id,
            'Class': display_class,
            'Centroid_X': cx,
            'Centroid_Y': cy,
            'Displacement_X': 0.0,
            'Displacement_Y': 0.0,
            'Orientation': det_data['orientation'],
            'Rotation_Delta': 0.0
        })

        min_row, min_col, max_row, max_col = det_data['bbox']
        cv2.rectangle(annotated_frame, (min_col, min_row), (max_col, max_row), (255, 0, 150), 1)
        cv2.putText(annotated_frame, f"NEW: {display_id}", (min_col, max(15, min_row - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 150), 1, cv2.LINE_AA)

    active_objects = updated_active_objects

    # --- DYNAMIC DISK WRITE ROUTER ---
    if EXPORT_MODE == "frames":
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), annotated_frame)
    else:
        # Lazy initialization of VideoWriter using the runtime shape of the processed frame
        if video_writer is None:
            height, width, _ = annotated_frame.shape
            video_path = os.path.join(run_dir, f"{file_slug}_tracked_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Exporting at 12.0 FPS for smooth microfluidic playback speeds
            video_writer = cv2.VideoWriter(video_path, fourcc, 12.0, (width, height))
        video_writer.write(annotated_frame)

# Clean up resources if video stream was initialized
if video_writer is not None:
    video_writer.release()

# ==========================================
# 5. MASTER TRACK DATASET EXPORT
# ==========================================
df = pd.DataFrame(all_frames_data)
csv_output_path = os.path.join(run_dir, "particle_tracks.csv")
df.to_csv(csv_output_path, index=False)

print("\n" + "=" * 60)
print("SUCCESS: Execution completed cleanly.")
print(f"-> Metrics Sheet: {csv_output_path}")
if EXPORT_MODE == "frames":
    print(f"-> Visual Assets: {total_frames} PNG frames inside {frames_dir}")
else:
    print(f"-> Visual Assets: Compressed MP4 Video saved to {video_path}")
print("=" * 60 + "\n")

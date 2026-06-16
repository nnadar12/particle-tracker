import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

def run_particle_tracker(tiff_path, export_mode, run_dir, file_slug):
    video_stack = tiff.imread(tiff_path)
    total_frames = len(video_stack)

    Y_MIN, Y_MAX = 48, 180
    X_MIN, X_MAX = 0, 511
    MAX_TRACKING_DISTANCE = 60
    EDGE_BUFFER_ZONE = 35

    cluster_counter, single_counter, internal_id_counter = 1, 1, 1
    active_objects, all_frames_data = {}, []

    video_writer = None
    frames_dir = None
    if export_mode == "frames":
        frames_dir = os.path.join(run_dir, "annotated_frames")
        os.makedirs(frames_dir, exist_ok=True)

    for frame_idx in range(total_frames):
        raw_frame = video_stack[frame_idx]
        cropped_frame = raw_frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

        thresh_value = threshold_otsu(cropped_frame)
        binary_frame = cropped_frame < thresh_value
        binary_frame[0:15, :] = False

        label_image = label(binary_frame)
        regions = regionprops(label_image)

        current_detections = []
        for prop in regions:
            if prop.area < 40: continue
            cy, cx = prop.centroid
            instant_class = "cluster" if (prop.area > 350 or prop.eccentricity > 0.60) else "single"
            current_detections.append({'centroid': (cx, cy), 'orientation': prop.orientation,
                                       'bbox': prop.bbox, 'instant_class': instant_class})

        img_normalized = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        annotated_frame = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
        frame_h, frame_w, _ = annotated_frame.shape

        matched_current_indices = set()
        updated_active_objects = {}

        # --- DATA ASSOCIATION ---
        if active_objects and current_detections:
            pairs = []
            for int_id, obj_data in active_objects.items():
                for det_idx, det_data in enumerate(current_detections):
                    dist = np.linalg.norm(np.array(obj_data['centroid']) - np.array(det_data['centroid']))
                    pairs.append((dist, int_id, det_idx))
            pairs.sort(key=lambda x: x[0])

            assigned_internal_ids = set()
            for dist, int_id, det_idx in pairs:
                if int_id in assigned_internal_ids or det_idx in matched_current_indices or dist > MAX_TRACKING_DISTANCE:
                    continue

                det_data = current_detections[det_idx]
                matched_current_indices.add(det_idx)
                assigned_internal_ids.add(int_id)
                obj_data = active_objects[int_id]
                cx, cy = det_data['centroid']

                # Logic: Calculate Rotation and Velocity
                d_theta = det_data['orientation'] - obj_data['orientation']
                if d_theta > np.pi / 2: d_theta -= np.pi
                elif d_theta < -np.pi / 2: d_theta += np.pi
                ang_vel = d_theta / 1.0 # delta t = 1ms

                # Upgrade logic
                if det_data['instant_class'] != obj_data['assigned_class'] or obj_data['final_id'] is None:
                    if det_data['instant_class'] == "cluster" and cx >= EDGE_BUFFER_ZONE:
                        obj_data['final_id'] = f"cluster{cluster_counter:02d}"; cluster_counter += 1
                    elif obj_data['final_id'] is None and cx >= EDGE_BUFFER_ZONE:
                        obj_data['final_id'] = f"single{single_counter:02d}"; single_counter += 1
                    obj_data['assigned_class'] = det_data['instant_class']

                # Store
                db_row_idx = len(all_frames_data)
                obj_data['database_indices'].append(db_row_idx)
                updated_active_objects[int_id] = {**det_data, 'final_id': obj_data['final_id'],
                                                  'assigned_class': obj_data['assigned_class'], 'database_indices': obj_data['database_indices']}

                all_frames_data.append({'Frame': frame_idx, 'Object_ID': obj_data['final_id'] or f"pending_{int_id}",
                                        'Class': obj_data['assigned_class'] or "pending", 'Centroid_X': cx, 'Centroid_Y': cy,
                                        'Orientation': det_data['orientation'], 'Rotation_Delta': d_theta, 'Angular_Velocity_rad_ms': ang_vel})

                # Graphics
                min_row, min_col, max_row, max_col = det_data['bbox']
                color = (0, 255, 0) if obj_data['assigned_class'] == "single" else (255, 0, 0)
                cv2.rectangle(annotated_frame, (min_col, min_row), (max_col, max_row), color, 1)
                cv2.putText(annotated_frame, f"{obj_data['final_id']}", (min_col, max(12, min_row-4)), 0, 0.35, color, 1, 2)
                cv2.putText(annotated_frame, f"X:{int(cx)} Y:{int(cy)}", (min_col, min(frame_h-12, max_row+11)), 0, 0.3, (0, 255, 255), 1, 2)
                cv2.putText(annotated_frame, f"w:{ang_vel:.3f}", (min_col, min(frame_h-2, max_row+21)), 0, 0.3, (0, 200, 255), 1, 2)

        # Handle New Entities
        for det_idx, det_data in enumerate(current_detections):
            if det_idx not in matched_current_indices:
                int_id = internal_id_counter; internal_id_counter += 1
                updated_active_objects[int_id] = {**det_data, 'final_id': None, 'assigned_class': None, 'database_indices': [len(all_frames_data)]}
                all_frames_data.append({'Frame': frame_idx, 'Object_ID': 'new', 'Class': 'new', 'Centroid_X': det_data['centroid'][0],
                                        'Centroid_Y': det_data['centroid'][1], 'Orientation': det_data['orientation'], 'Rotation_Delta': 0.0, 'Angular_Velocity_rad_ms': 0.0})

        active_objects = updated_active_objects

        # Write Output
        if export_mode == "frames":
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), annotated_frame)
        else:
            if video_writer is None:
                video_writer = cv2.VideoWriter(os.path.join(run_dir, f"{file_slug}_tracked_output.mp4"),
                                               cv2.VideoWriter_fourcc(*'mp4v'), 12.0, (frame_w, frame_h))
            video_writer.write(annotated_frame)

    if video_writer: video_writer.release()
    pd.DataFrame(all_frames_data).to_csv(os.path.join(run_dir, "particle_tracks.csv"), index=False)

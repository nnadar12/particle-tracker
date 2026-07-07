import os
import cv2
import numpy as np
import pandas as pd
import tifffile

def run_cluster_tracker(input_path, output_path):
    # Channel Bounds based on user coordinates
    Y_MIN, Y_MAX = 0, 416
    X_MIN, X_MAX = 0, 184

    # --- CLUSTER FILTER CONSTRAINTS (Mirrored from Diagnostics) ---
    MIN_CLUSTER_AREA = 96
    MAX_CLUSTER_AREA = 180
    MIN_CLUSTER_ASPECT = 1.10
    MAX_CLUSTER_ASPECT = 1.7
    MIN_CLUSTER_RADIUS = 6.5
    MAX_CLUSTER_RADIUS = 12.0

    print(f"\n[+] Executing Clean Target-Mirrored Cluster Tracker...")
    print(f"[+] Input File: {input_path}")
    print(f"[+] Output File: {output_path}")

    cluster_records = []
    csv_output_path = os.path.splitext(output_path)[0] + "_clusters_detected.csv"

    with tifffile.TiffFile(input_path) as tif:
        total_frames = len(tif.pages)

        # --- STEP 1: GENERATE BACKGROUND MODEL ---
        print("[+] Generating temporal background model to eliminate static artifacts...")
        bg_samples = []
        sample_step = max(1, total_frames // 50)
        for idx in range(0, total_frames, sample_step):
            raw_img = tif.pages[idx].asarray()
            bg_samples.append(raw_img[Y_MIN:Y_MAX, X_MIN:X_MAX])
        temporal_median_bg = np.median(bg_samples, axis=0).astype(np.uint8)

        # --- STEP 2: PROCESS ALL FRAMES AND WRITE TO TIFF ---
        with tifffile.TiffWriter(output_path, bigtiff=True) as out_tif:
            for frame_idx in range(total_frames):
                page = tif.pages[frame_idx]
                raw_frame = page.asarray()
                roi = raw_frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

                # Exact Image Processing Pipeline from inspect_target_frames
                equalized = cv2.subtract(temporal_median_bg, roi)
                norm_equalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                smoothed = cv2.GaussianBlur(norm_equalized, (3, 3), 0)
                _, thresh = cv2.threshold(smoothed, 35, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Setup Display Canvas
                display_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                canvas = cv2.cvtColor(display_roi, cv2.COLOR_GRAY2BGR)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 15:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h != 0 else 0
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                    # Evaluate against your exact geometric criteria
                    is_cluster = (MIN_CLUSTER_AREA <= area <= MAX_CLUSTER_AREA and
                                  MIN_CLUSTER_ASPECT <= aspect_ratio <= MAX_CLUSTER_ASPECT and
                                  (MIN_CLUSTER_RADIUS <= radius <= MAX_CLUSTER_RADIUS))

                    if is_cluster:
                        # Annotate the cluster directly on the frame canvas
                        cluster_r = int(radius + 3)
                        cv2.circle(canvas, (int(cx), int(cy)), cluster_r, (0, 0, 255), 2)
                        cv2.putText(canvas, "CLUSTER", (int(cx) - 22, int(cy) - cluster_r - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                        ratio = (area/(radius*radius*3.14))*aspect_ratio

                        print(f"\n[!] Cluster Identified | Frame {frame_idx} | Area: {area:.1f} | AR: {aspect_ratio:.2f} | Radius: {radius:.1f} | ratio: {ratio:.2f}")

                        # Save coordinates for data export
                        cluster_records.append({
                            'Frame': frame_idx,
                            'Cluster_X': round(cx, 2),
                            'Cluster_Y': round(cy, 2)
                        })

                # Append the annotated frame canvas to the output multi-page TIFF file
                out_tif.write(canvas)
                print(f" -> Tracking flow sequence: [{frame_idx + 1}/{total_frames}]", end="\r")

    # --- STEP 3: EXPORT DATA SUMMARY ---
    if cluster_records:
        pd.DataFrame(cluster_records).to_csv(csv_output_path, index=False)
        print(f"\n\n[+] Tracking complete! Log written to: {csv_output_path}")
    else:
        print("\n\n[-] Sequence complete. No clusters cleared the strict criteria matrix.")

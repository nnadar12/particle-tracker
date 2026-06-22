import os
import cv2
import numpy as np
import tifffile

def run_cell_tracker(input_path, output_path):
    # Channel Bounds based on user coordinates
    Y_MIN, Y_MAX = 0, 416
    X_MIN, X_MAX = 0, 184

    print(f"\n[+] Executing Phase 1 & 2: Bottom-Hat + Geometric Contour Isolation...")
    print(f"[+] Input File: {input_path}")
    print(f"[+] Output File: {output_path}")

    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))

    with tifffile.TiffFile(input_path) as tif:
        total_available = len(tif.pages)
        frames_to_process = min(20, total_available)
        print(f"[+] Processing {frames_to_process} frames...")

        with tifffile.TiffWriter(output_path, bigtiff=True) as out_tif:
            for frame_idx in range(frames_to_process):
                page = tif.pages[frame_idx]
                raw_frame = page.asarray()

                # --- PHASE 1: BOTTOM-HAT EQUALIZATION ---
                roi = raw_frame[Y_MIN:Y_MAX, X_MIN:X_MAX]
                background_map = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, bg_kernel)
                equalized_roi = cv2.subtract(background_map, roi)

                # FIX 1: Normalize the equalized image to a strict 8-bit scale (0-255).
                # This ensures our threshold of 35 works perfectly regardless of the TIF bit-depth.
                norm_equalized = cv2.normalize(equalized_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # --- PHASE 2: GEOMETRIC ISOLATION ---
                smoothed = cv2.GaussianBlur(norm_equalized, (3, 3), 0)
                _, thresh = cv2.threshold(smoothed, 35, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # FIX 2: Normalize the drawing canvas to 8-bit so the green circles are actually visible
                display_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                annotated_roi = cv2.cvtColor(display_roi, cv2.COLOR_GRAY2BGR)

                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    # FIX 3: Increased max area to 1000 to mathematically accommodate cells with a radius up to ~17
                    if 15 < area < 1000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = float(w) / h

                        if 0.5 <= aspect_ratio <= 1.8:
                            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                            r = int(radius)

                            if 4 <= r <= 20:
                                # Draw a precise green tracking ring and red center coordinate dot
                                cv2.circle(annotated_roi, (int(cx), int(cy)), r, (0, 255, 0), 1)
                                cv2.circle(annotated_roi, (int(cx), int(cy)), 2, (0, 0, 255), -1)

                out_tif.write(annotated_roi)
                print(f" -> Processed frame [{frame_idx + 1}/{frames_to_process}]", end="\r")

    print(f"\n\n[+] Geometric Isolation Execution Complete.")
    print(f"-> Please open and verify the output file: {output_path}\n")

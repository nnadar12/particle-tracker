import os
import sys
from datetime import datetime
import particle_tracker
import cell_tracker

DATA_DIR = "../data"
OUTPUT_DIR = "../output"

def get_file_selection():
    valid_extensions = ('.tif', '.tiff')
    tiff_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(valid_extensions)]

    if not tiff_files:
        print(f"Error: No files found in '{DATA_DIR}'.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("         MULTI-OBJECT MICROSCOPY TRACKER SYSTEM           ")
    print("=" * 60)
    for idx, filename in enumerate(tiff_files, 1):
        print(f"  [{idx}] {filename}")
    print("=" * 60)

    while True:
        try:
            choice = input("Select a file number (or 'q' to quit): ").strip()
            if choice.lower() == 'q': sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(tiff_files):
                return tiff_files[idx]
            print(f"Choose 1-{len(tiff_files)}.")
        except ValueError:
            print("Invalid entry.")

def main():
    print("[1] Particle Tracker")
    print("[2] Cell/Cluster Tracker (Phase 1: Equalization Test)")
    algo_choice = input("Select tracker algorithm: ")

    selected_file = get_file_selection()
    input_path = os.path.join(DATA_DIR, selected_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_slug = os.path.splitext(selected_file)[0]

    # Route to Particle Tracker
    if algo_choice == '1':
        print("\n[1] Individual PNG Frames\n[2] Compiled MP4 Video")
        mode_choice = input("Select export mode [1 or 2]: ")
        export_mode = "frames" if mode_choice == '1' else "video"

        run_dir = os.path.join(OUTPUT_DIR, f"run_{file_slug}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        particle_tracker.run_particle_tracker(
            input_path, export_mode, run_dir, file_slug
        )

    # Route to Cell Tracker (Phase 1 Test)
    elif algo_choice == '2':
        output_path = os.path.join(OUTPUT_DIR, f"phase1_equalized_{file_slug}_{timestamp}.tif")
        cell_tracker.run_cell_tracker(input_path, output_path)

if __name__ == "__main__":
    main()

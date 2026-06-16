import os
import sys
from datetime import datetime
import particle_tracker

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
    algo_choice = input("Select tracker algorithm: ")

    selected_file = get_file_selection()

    print("\n[1] Individual PNG Frames\n[2] Compiled MP4 Video")
    mode_choice = input("Select export mode [1 or 2]: ")
    export_mode = "frames" if mode_choice == '1' else "video"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_slug = os.path.splitext(selected_file)[0]
    run_dir = os.path.join(OUTPUT_DIR, f"run_{file_slug}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    if algo_choice == '1':
        particle_tracker.run_particle_tracker(
            os.path.join(DATA_DIR, selected_file),
            export_mode, run_dir, file_slug
        )

if __name__ == "__main__":
    main()

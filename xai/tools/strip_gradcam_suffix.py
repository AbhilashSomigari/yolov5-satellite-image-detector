#!/usr/bin/env python3
"""Strip a ".gradcam" infix from every filename in a folder (cleanup for
prediction/label files that picked up the suffix during batch CAM runs).

Usage:
    python xai/tools/strip_gradcam_suffix.py data/pred_boxes
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="data/pred_boxes", help="Folder to clean up")
    args = parser.parse_args()

    for filename in os.listdir(args.folder):
        if ".gradcam" in filename:
            old_path = os.path.join(args.folder, filename)
            new_filename = filename.replace(".gradcam", "")
            new_path = os.path.join(args.folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

    print("Done! All '.gradcam' removed from filenames.")


if __name__ == "__main__":
    main()

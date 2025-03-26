"""
Script to change YOLOv8 class_id into integer
"""
import os
import argparse


def fix_class_id(labels_dir):
    """
    Parses YOLOv8 label files in a directory and corrects class ID to integer.

    Args:
      labels_dir: Path to the directory containing the label files.
    """
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()

            with open(filepath, "w") as f:
                for line in lines:
                    parts = line.strip().split()
                if len(parts) == 5:  # Ensure it's a valid YOLO label line
                    parts[0] = str(int(float(parts[0])))  # Convert to int and back to string
                    f.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix class ID in YOLOv8 label files.")
    parser.add_argument("train_labels_dir", help="Path to the training labels directory.")
    parser.add_argument("valid_labels_dir", help="Path to the validation labels directory.")
    args = parser.parse_args()

    fix_class_id(args.train_labels_dir)
    fix_class_id(args.valid_labels_dir)

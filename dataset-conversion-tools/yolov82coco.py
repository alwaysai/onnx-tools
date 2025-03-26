import os
import json
import yaml
from PIL import Image


def yolov8_to_coco(data_yaml_path, output_dir, num_samples=None):
    """
    Converts YOLOv8 labels to COCO JSON format with category ID re-indexing,
    handling of negative bounding box coordinates, and rounding of coordinates
    and area to 4 significant figures. Handles multiple annotations per image
    and images with empty label files.

    Args:
        data_yaml_path: Path to the data.yaml file.
        output_dir: Directory to save the output JSON files (train.json and val.json).
        num_samples: Number of images to sample from each split (default: None, processes all images).
    """

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    for split in ['train', 'val']:
        images = []
        annotations = []
        categories = [{'id': i + 1, 'name': name} for i, name in enumerate(data['names'])]
        ann_id = 0

        img_dir = os.path.join(data['path'], data[split])
        labels_dir = os.path.join(data['path'], data[split].replace('images', 'labels'))

        img_count = 0
        for filename in os.listdir(img_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_id = len(images)
                img_path = os.path.join(img_dir, filename)
                img_width, img_height = get_image_size(img_path)

                images.append({
                    'id': img_id,
                    'file_name': filename,
                    'width': img_width,
                    'height': img_height
                })

                name_without_ext = os.path.splitext(filename)[0]
                label_file = os.path.join(labels_dir, name_without_ext + '.txt')

                if os.path.exists(label_file):
                    with open(label_file, 'r') as lf:
                        for line in lf:
                            # Skip empty lines
                            if not line.strip():
                                continue

                            try:
                                cls_id, x_center, y_center, w, h = map(float, line.strip().split())
                                x = (x_center - w / 2) * img_width
                                y = (y_center - h / 2) * img_height
                                width = w * img_width
                                height = h * img_height

                                x = max(0, x)
                                y = max(0, y)

                                annotations.append({
                                    'id': ann_id,
                                    'image_id': img_id,
                                    'category_id': int(cls_id) + 1,
                                    'bbox': [round(x, 4), round(y, 4), round(width, 4), round(height, 4)],
                                    'area': round(width * height, 4),
                                    'iscrowd': 0
                                })
                                ann_id += 1
                            except ValueError as e:
                                print(f"Error processing file: {label_file}")
                                print(f"Original error message: {e}")

            img_count += 1
            if num_samples and img_count >= num_samples:
                break

        coco_data = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        output_json_path = os.path.join(output_dir, f'{split}.json')
        with open(output_json_path, 'w') as f:
            json.dump(coco_data, f, indent=4)


def get_image_size(img_path):
    """
    Helper function to get image width and height.

    Args:
        img_path: Path to the image.

    Returns:
        tuple: (width, height)
    """
    with Image.open(img_path) as img:
        return img.width, img.height


if __name__ == "__main__":
    data_yaml_path = 'data.yaml'
    output_dir = 'annotations'
    yolov8_to_coco(data_yaml_path, output_dir)

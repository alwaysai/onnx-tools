import os
import json
import yaml
from PIL import Image


def yolov8_to_coco_sample(data_yaml_path, output_dir, num_samples=5):
    """
    Converts YOLOv8 labels to COCO JSON format for the first 'num_samples'
    images in train and val sets, with category ID re-indexing and handling
    of negative bounding box coordinates. Rounds coordinates and area to 4
    significant figures.

    Args:
        data_yaml_path: Path to the data.yaml file.
        output_dir: Directory to save the output JSON files (train.json and val.json).
        num_samples: Number of images to sample from each split (default: 5).
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

        # Sample the first 'num_samples' images
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

                # Extract filename without extension for label file matching
                name_without_ext = os.path.splitext(filename)[0]
                label_file = os.path.join(labels_dir, name_without_ext + '.txt')

                if os.path.exists(label_file):
                    with open(label_file, 'r') as lf:
                        for line in lf:
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

                img_count += 1
                if img_count >= num_samples:
                    break  # Exit the loop after processing 'num_samples' images

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
    data_yaml_path = 'data.yaml'  # Path to your data.yaml
    output_dir = 'coco_dataset_sample'  # Output directory for JSON files
    yolov8_to_coco_sample(data_yaml_path, output_dir)

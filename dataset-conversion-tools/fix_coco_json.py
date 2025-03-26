import os
import json

def fix_coco_json(json_dir):
    """
    Fixes COCO JSON files by re-indexing category IDs to start from 1 and 
    ensuring bounding box coordinates are not negative.

    Args:
        json_dir: Directory containing the COCO JSON files.
    """

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r') as f:
                coco_data = json.load(f)

            # Re-index category IDs in 'categories'
            for cat in coco_data['categories']:
                cat['id'] += 1  

            for ann in coco_data['annotations']:
                # Re-index category IDs in 'annotations'
                ann['category_id'] += 1  

                # Ensure bounding box coordinates are not negative
                x, y, w, h = ann['bbox']
                ann['bbox'] = [max(0, x), max(0, y), w, h] 

            with open(filepath, 'w') as f:
                json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    json_dir = 'Annotations'  # Replace with your directory
    fix_coco_json(json_dir)

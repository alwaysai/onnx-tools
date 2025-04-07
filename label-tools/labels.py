import json

# Path to your COCO annotations file (train or val)
coco_json = "annotations/train.json"  # or val.json

# Load the COCO annotations
with open(coco_json, "r") as f:
    data = json.load(f)

# Extract categories and sort by their 'id'
categories = sorted(data["categories"], key=lambda x: x["id"])
class_names = [cat["name"] for cat in categories]

# Save to a label file
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ Saved class names to labels.txt in correct order.")

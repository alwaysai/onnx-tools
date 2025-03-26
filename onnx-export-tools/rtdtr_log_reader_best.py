import json


def find_best_epoch(log_file):
    """
    Parses a log file and returns the epoch with the highest AP.

    Args:
        log_file: Path to the log file.

    Returns:
        A tuple containing the best epoch and its corresponding AP.
    """
    best_epoch = 0
    best_ap = 0

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                epoch = data['epoch']
                ap = data['test_coco_eval_bbox'][0]  # Extract AP

                if ap > best_ap:
                    best_ap = ap
                    best_epoch = epoch
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")

    return best_epoch, best_ap


# Example usage
log_file = 'your_log_file.log'  # Replace with your log file path
best_epoch, best_ap = find_best_epoch(log_file)
print(f"Best epoch: {best_epoch}, AP: {best_ap}")

import json
import argparse


def analyze_operator_durations(profiling_file):
    """
    Analyzes an ONNX Runtime profiling JSON file to calculate the total
    duration of different operators.

    Args:
        profiling_file (str): Path to the JSON profiling file.
    """
    operator_durations = {}  # Dictionary to store durations by operator name

    with open(profiling_file, 'r') as f:
        data = json.load(f)

    # Calculate total duration only once
    total_duration = sum(event.get('dur', 0) for event in data)

    for event in data:
        if 'args' in event and 'op_name' in event['args']:
            op_name = event['args']['op_name']
            operator_durations[op_name] = operator_durations.get(op_name, 0) + event['dur']

    # Print durations and percentages
    total_duration /= 1000  # Convert to milliseconds
    print("Operator Durations:")
    for op_name, duration in operator_durations.items():
        duration_ms = duration / 1000
        percentage = (duration / total_duration) * 100 if total_duration else 0
        print(f"  {op_name}: {duration_ms:.4f} ms ({percentage:.2f}%)")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze ONNX Runtime profiling data")
    parser.add_argument("profiling_file", type=str, help="Path to the JSON profiling file")
    args = parser.parse_args()

    analyze_operator_durations(args.profiling_file)

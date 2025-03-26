# python onnx_output_shape_check_v7.py --model my_model.onnx --batch_sizes 1 2 4 8 --height 640 --width 640
import onnxruntime as ort
import numpy as np
import time
import argparse
import torch


def print_model_info(session):
    """Print input and output details of the ONNX model."""
    print("Model Inputs:")
    for input in session.get_inputs():
        print(f"  {input.name}: shape={input.shape}, type={input.type}")

    print("Model Outputs:")
    for output in session.get_outputs():
        print(f"  {output.name}: shape={output.shape}, type={output.type}")


def run_inference(session, input_data):
    """Run inference on the ONNX model and measure time."""
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    elapsed_time = time.time() - start_time
    return outputs, elapsed_time


def main(args):
    # Print ONNX Runtime information
    print("ONNX Runtime version:", ort.__version__)
    available_providers = ort.get_available_providers()
    print("Available providers:", available_providers)

    if args.device == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            provider = "CUDAExecutionProvider"
        else:
            print("CUDA is not available. Falling back to CPUExecutionProvider.")
            provider = "CPUExecutionProvider"
    elif args.device == "cpu":
        provider = "CPUExecutionProvider"
    else:
        raise ValueError(f"Unsupported device '{args.device}'. Use 'cpu' or 'cuda'.")

    # Enable verbose logging
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 0  # Set log level to verbose for detailed logging
    session_options.enable_profiling = True

    # Enable pinned memory for CUDA (only if CUDA is being used)
    if provider == "CUDAExecutionProvider":
        session_options.add_session_config_entry('CUDAExecutionProvider.do_copy_in_default_stream', 'true')

    # Load ONNX model
    print(f"Using provider: {provider}")
    session = ort.InferenceSession(args.model, session_options, providers=[provider])

    # Print model details
    print_model_info(session)

    # Test multiple batch sizes
    for batch_size in args.batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")

        # Enable CUDA Graph with dynamic batch size
        if provider == "CUDAExecutionProvider":
            session_options.add_session_config_entry('CUDAExecutionProvider.enable_cuda_graph', '1')
            session_options.add_session_config_entry('CUDAExecutionProvider.cuda_graph_batch_size', str(batch_size))

            # Re-create the session with the updated options
            session = ort.InferenceSession(args.model, session_options, providers=[provider])

        # Generate input data (on GPU if CUDA is selected)
        if args.device == "cuda":
            app_start_time = time.time()
            input_data = torch.randn(batch_size, 3, args.height, args.width, device='cuda').cpu().numpy()
        else:
            app_start_time = time.time()
            input_data = np.random.randn(batch_size, 3, args.height, args.width).astype(np.float32)

        # Warm-up (if enabled)
        if args.warmup:
            print("Warm-up phase...")
            for _ in range(5):
                run_inference(session, input_data)

        try:
            # Multiple Inference Runs to Amortize Cost
            num_runs = 10
            inference_times = []
            for _ in range(num_runs):
                outputs, inference_time = run_inference(session, input_data)
                inference_times.append(inference_time)

            print(f"Inference successful for batch size {batch_size}!")
            print(f"Output shape: {outputs[0].shape}")
            print("Inference times:", inference_times)

            # Calculate and print averages
            avg_inference_time_all = np.mean(inference_times)
            avg_inference_time_excl_first = np.mean(inference_times[1:])  # Exclude first run
            print(f"Average inference time (all runs): {avg_inference_time_all:.4f} seconds")
            print(f"Average inference time (excluding first run): {avg_inference_time_excl_first:.4f} seconds")

            # Calculate and print application execution time
            app_elapsed_time = time.time() - app_start_time
            print(f"Application Execution Time: {app_elapsed_time:.4f} seconds")

        except Exception as e:
            print(f"Inference failed for batch size {batch_size}: {e}")

    # End profiling and save profiling data
    profile_file = session.end_profiling()
    print(f"Profiling data saved to: {profile_file}")
    print("\nTo analyze profiling data, open the JSON file and examine node execution assignments.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test ONNX model with ONNX Runtime")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1], help="List of batch sizes to test")
    parser.add_argument("--height", type=int, default=640, help="Input image height")
    parser.add_argument("--width", type=int, default=640, help="Input image width")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for inference (cpu or cuda). Default: cpu")
    parser.add_argument("--warmup", action="store_true", help="Enable warm-up phase")
    args = parser.parse_args()

    # Call main with parsed arguments
    main(args)

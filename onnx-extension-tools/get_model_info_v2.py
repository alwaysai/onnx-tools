"""
Get ONNX model input and output information for all tensors
"""
import onnx
from onnx import shape_inference
from pathlib import Path
from typing import Tuple, Dict, Any, List  # noqa: F401
import argparse


def get_model_info(input_model_path: Path) -> Dict[str, Any]:
    """
    Retrieves information about an ONNX model, including input and all output shapes.

    Args:
        input_model_path (Path): Path to the ONNX model file.

    Returns:
        Dict[str, Any]: A dictionary containing input and output information:
            - "input_shape": Tuple[Any, ...] representing the input shape.
            - "output_shapes": Dict[str, Tuple[Any, ...]] where keys are output names
                              and values are their corresponding shapes.
            - "batch_dimension": The symbolic name, -1, or value of the batch dimension.
    Raises:
        FileNotFoundError: If the input_model_path does not exist.
        onnx.onnx_cpp2py_export.checker.ValidationError: If the onnx model is invalid.
    """
    if not input_model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {input_model_path}")

    print("Loading the model...")
    model = onnx.load(str(input_model_path.resolve(strict=True)))

    print("Inferring model shapes...")
    model_with_shape_info = shape_inference.infer_shapes(model)

    model_input = model_with_shape_info.graph.input[0]
    model_input_shape = model_input.type.tensor_type.shape

    def get_dim_value(dim):
        if dim.dim_param:
            return dim.dim_param  # Dynamic dimension (symbolic name)
        elif dim.dim_value >= 0:
            return dim.dim_value  # Static dimension
        else:
            return -1  # Dynamic dimension (-1)

    input_shape = tuple(get_dim_value(dim) for dim in model_input_shape.dim)

    # Get information for all output tensors
    output_shapes = {}
    for model_output in model_with_shape_info.graph.output:
        output_name = model_output.name
        output_shape = model_output.type.tensor_type.shape
        output_shape_tuple = tuple(get_dim_value(dim) for dim in output_shape.dim)
        output_shapes[output_name] = output_shape_tuple

    batch_dimension = get_dim_value(model_input_shape.dim[0]) if model_input_shape.dim else None

    return {
        "input_shape": input_shape,
        "output_shapes": output_shapes,
        "batch_dimension": batch_dimension,
    }


def main(args: argparse.Namespace):
    """
    Main function to get model info and print it.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        model_info = get_model_info(args.model_path)
        print("Model Information:")
        for key, value in model_info.items():
            if key == "output_shapes":
                print("  Output Shapes:")
                for output_name, output_shape in value.items():
                    print(f"    {output_name}: {output_shape}")
            else:
                print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print(f"Error: ONNX model is invalid: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get ONNX model information.")
    parser.add_argument("model_path", type=Path, help="Path to the ONNX model file.")
    args = parser.parse_args()
    main(args)

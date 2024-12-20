import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import onnx

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


class PostProcessor(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def forward(self, x):
        boxes = x['pred_boxes']
        boxes[:, :, [0, 2]] *= self.img_size[1]
        boxes[:, :, [1, 3]] *= self.img_size[0]
        scores, classes = torch.max(x['pred_logits'], dim=2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    """Suppress common warnings during export."""
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_model(weights, cfg_file):
    """Load the PyTorch model."""
    cfg = YAMLConfig(cfg_file, resume=weights)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)

    # Check if 'ema' exists in checkpoint and load state accordingly
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    return cfg.model.deploy()


def export_onnx_model(model, img_size, batch_size, onnx_output_file, opset_version, dynamic_axes, simplify):
    """Export the PyTorch model to ONNX format."""
    device = torch.device('cpu')

    # Move the model to CPU
    model = model.to(device)

    # Determine if the batch size is dynamic or fixed
    symbolic_batch_size = "batch_size" if dynamic_axes else batch_size

    # Create a sample input for export (with symbolic or fixed batch size)
    dummy_input = torch.randn(2 if dynamic_axes else batch_size, 3, *img_size).to(device)

    # Input and output names
    input_names = ['input']
    output_names = ['boxes', 'scores', 'classes']

    # Dynamic axes configuration
    dynamic_axes_dict = {
        'input': {0: symbolic_batch_size},  # Mark batch dimension as dynamic
        'boxes': {0: symbolic_batch_size},
        'scores': {0: symbolic_batch_size},
        'classes': {0: symbolic_batch_size}
    }

    # Log export configuration
    if dynamic_axes:
        print(f"Exporting a dynamic ONNX model with batch size: {symbolic_batch_size}.")
    else:
        print(f"Exporting a static ONNX model with fixed batch size: {batch_size}.")

    # Export the model to ONNX
    print(f'Exporting ONNX model to {onnx_output_file}...')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_file,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict if dynamic_axes else None
    )
    print(f'Exported ONNX model to {onnx_output_file}')

    # Infer shapes for validation
    infer_onnx_shapes(onnx_output_file)

    # Simplify the model (if requested)
    if simplify:
        simplify_onnx_model(onnx_output_file, dynamic_axes, dummy_input)


def simplify_onnx_model(onnx_output_file, dynamic_axes, dummy_input):
    """Simplify the ONNX model."""
    import onnxsim

    print(f'Simplifying ONNX model at {onnx_output_file}...')
    test_input_shapes = {'input': list(dummy_input.shape)} if dynamic_axes else None
    simplified_model, check = onnxsim.simplify(
        onnx_output_file,
        test_input_shapes=test_input_shapes
    )
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(simplified_model, onnx_output_file)
    print(f'Simplified ONNX model saved to {onnx_output_file}')


def infer_onnx_shapes(onnx_output_file):
    """Infer shapes and validate the ONNX model."""
    model = onnx.load(onnx_output_file)
    inferred_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred_model, onnx_output_file)
    print(f"Inferred shapes and saved updated ONNX model to {onnx_output_file}")


def ensure_dynamic_support(model):
    """
    Modify the model if needed to ensure it supports dynamic shapes.

    This step adjusts operations that may assume static shapes, like reshaping.
    """
    class DynamicWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Ensure dynamic batch size compatibility in the model
            output = self.model(x)
            for key in output:
                if key == 'pred_boxes':
                    # Dynamically adjust the shape of boxes if needed
                    output[key] = output[key].reshape(x.shape[0], -1, 4)  # Dynamic batch size
                elif key == 'pred_logits':
                    output[key] = output[key].reshape(x.shape[0], -1, output[key].shape[-1])
            return output

    return DynamicWrapper(model)


def main(args):
    """Main function to handle the script logic."""
    suppress_warnings()
    print(f"DEBUG: args.dynamic={args.dynamic}")  # Debug dynamic switch
    model = load_model(args.weights, args.config)
    img_size = args.size * 2 if len(args.size) == 1 else args.size
    model = nn.Sequential(model, PostProcessor(img_size))
    export_onnx_model(model, img_size, args.batch, args.output, args.opset, args.dynamic, args.simplify)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--config', required=True, help='Input YAML (.yml) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch-size')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output ONNX model file name')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.config):
        raise SystemExit('Invalid config file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at the same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
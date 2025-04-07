import os
import sys
import argparse
import warnings

import torch
import torch.nn as nn
import onnx
import onnxruntime_extensions.tools.pre_post_processing as pre_post
from onnx import TensorProto  # Import TensorProto for data types

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Make sure this import works correctly
from src.core import YAMLConfig  # noqa: E402


class PostProcessor(nn.Module):
    """
    Post-processes the model output to get the final detections.
    """

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
    torch_version = torch.__version__.split('+')[0]
    if torch_version <= "2.0.1":
        checkpoint = torch.load(weights, map_location='cpu')
    else:
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

    if args.dynamic:
        model = ensure_dynamic_support(model)  # Call the function for dynamic batch size

    export_onnx_model(model, img_size, args.batch, args.output, args.opset, args.dynamic, args.simplify)

    # Load the exported ONNX model
    onnx_model = onnx.load(args.output)

    # Define pre-processing pipeline
    inputs = [pre_post.create_named_value("image_bytes", onnx.TensorProto.UINT8, ["batch", "num_bytes"])]
    pipeline = pre_post.PrePostProcessor(inputs, args.opset)

    pre_processing_steps = [
        pre_post.ConvertImageToBGR(name="BGRImageHWC"),
        pre_post.ChannelsLastToChannelsFirst(name="BGRImageCHW"),
        pre_post.Resize((img_size[0], img_size[1]), policy='not_larger', layout='CHW', name="Resize"),
        pre_post.LetterBox(target_shape=(img_size[0], img_size[1]), layout='CHW', name="LetterBox"),
        pre_post.ImageBytesToFloat(),
    ]

    pipeline.add_pre_processing(pre_processing_steps)

    # Define post-processing pipeline (RT-DETR specific)
    post_processing_steps = [
        # Confidence Filtering
        pre_post.Step(
            name="Greater",
            op_type="Greater",
            input_names=["scores", "confidence_threshold"],
            output_names=["mask"],
            attrs={"to_add": [
                ("value", onnx.AttributeProto.AttributeType.FLOAT, args.confidence_threshold)
            ]},
        ),
        pre_post.Step(
            name="GatherElements_boxes",
            op_type="Gather",
            input_names=["boxes", "mask"],
            output_names=["filtered_boxes"],
            attrs={"to_add": [("axis", onnx.AttributeProto.AttributeType.INT, 1)]},
        ),
        pre_post.Step(
            name="GatherElements_scores",
            op_type="Gather",
            input_names=["scores", "mask"],
            output_names=["filtered_scores"],
            attrs={"to_add": [("axis", onnx.AttributeProto.AttributeType.INT, 1)]},
        ),
        pre_post.Step(
            name="GatherElements_classes",
            op_type="Gather",
            input_names=["classes", "mask"],
            output_names=["filtered_classes"],
            attrs={"to_add": [("axis", onnx.AttributeProto.AttributeType.INT, 1)]},
        ),
        # Scale boxes back to original image
        (pre_post.ScaleNMSBoundingBoxes(layout='CHW'),
            [
                # A connection from original image to input 1
                # A connection from the resized image to input 2
                # A connection from the LetterBoxed image to input 3
                # We use the three images to calculate the scale factor and offset.
                # With scale and offset, we can scale the bounding box back to the original image.
                pre_post.utils.IoMapEntry("BGRImageHWC", producer_idx=0, consumer_idx=1),
                pre_post.utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
                pre_post.utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
        ]),
    ]

    pipeline.add_post_processing(post_processing_steps)

    # Add the confidence_threshold input to the model
    confidence_threshold_input = pre_post.create_named_value(
        name="confidence_threshold",
        data_type=TensorProto.FLOAT,
        shape=[1],  # Scalar value
    )
    onnx_model.graph.input.extend([confidence_threshold_input])

    # Run the pipeline and save the modified model
    modified_model = pipeline.run(onnx_model)
    onnx.save(modified_model, args.output)

    print(f"ONNX model with pre/post-processing saved to {args.output}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--config', required=True, help='Input YAML (.yml) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch-size')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output ONNX model file name')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold')
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

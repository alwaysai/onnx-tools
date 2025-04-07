"""
Conversion script that takes yolov8 pose estimation model to ONNX format
and adds pre and post processings extensions with dynamic batching.

Changes to version 3
1. The model.export() call now includes dynamic=True. This tells Ultralytics to
   export the ONNX model with dynamic axes, allowing for variable batch sizes.
2. The inputs definition within PrePostProcessor is modified to
   inputs = [create_named_value("image_bytes", onnx.TensorProto.UINT8, ["batch", "num_bytes"])].
   This sets the input shape to ["batch", "num_bytes"], where "batch" will be dynamic.
3. The Unsqueeze([0]) operation within the preprocessing steps has been removed. Because
   the input now already has a dynamic batch dimension, this step is unnecessary.
"""

import argparse
import onnx
import onnxruntime_extensions  # noqa: F401
from onnxruntime_extensions.tools.pre_post_processing import (
    PrePostProcessor,
    create_named_value,
    ConvertImageToBGR,
    ChannelsLastToChannelsFirst,
    Resize,
    LetterBox,
    ImageBytesToFloat,
    Squeeze,
    Transpose,
    Split,
    SelectBestBoundingBoxesByNMS,
    ScaleNMSBoundingBoxesAndKeyPoints,
    utils,
)
from pathlib import Path
import ultralytics


def generate_model(
    pt_model_path: Path,
    onnx_model_path: Path,
    model_size: int = 640,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.25,
):
    """
    Generates an ONNX model with pre and post-processing steps and dynamic batching.

    Args:
        pt_model_path (Path): Path to the PyTorch model.
        onnx_model_path (Path): Path to save the ONNX model.
        model_size (int): Size of the model input (default: 640).
        iou_threshold (float): IoU threshold for NMS (default: 0.7).
        score_threshold (float): Score threshold for NMS (default: 0.25).
    """
    # Load the PyTorch model
    model = ultralytics.YOLO(str(pt_model_path))

    # Export to ONNX with dynamic axes
    success = model.export(format="onnx", opset=18, dynamic=True)
    assert success, "Failed to export model to ONNX"

    # Load the ONNX model
    model = onnx.load(str(onnx_model_path.resolve(strict=True)))

    # Add pre-processing and post-processing steps
    onnx_opset = 18
    # create a ValueInfoProto object in ONNX.  A ValueInfoProto describes a tensor, including its name,
    # data type, and shape.
    inputs = [create_named_value("image_bytes", onnx.TensorProto.UINT8, ["batch", "num_bytes"])]  # dynamic batching here.
    pipeline = PrePostProcessor(inputs, onnx_opset)

    pre_processing_steps = [
        ConvertImageToBGR(name="BGRImageHWC"),  # jpg/png image to BGR in HWC layout
        ChannelsLastToChannelsFirst(name="BGRImageCHW"),  # HWC to CHW
        # Resize to match model input. Uses not_larger as we use LetterBox to pad as needed.
        Resize((model_size, model_size), policy='not_larger', layout='CHW'),
        LetterBox(target_shape=(model_size, model_size), layout='CHW'),  # padding or cropping the image to (model_size, model_size) # noqa: E501
        ImageBytesToFloat(),  # Convert to float in range 0..1
        # Unsqueeze([0]),  # add batch, CHW --> 1CHW. Removed as batch is already dynamic.
    ]

    pipeline.add_pre_processing(pre_processing_steps)

    # NonMaxSuppression and drawing boxes
    post_processing_steps = [
        Squeeze([0]),  # Squeeze to remove batch dimension from [batch, 56, 8200] output
        Transpose([1, 0]),  # reverse so result info is inner dim
        # split the 56 elements into the box, score for the 1 class, and mask info (17 locations x 3 values)
        Split(num_outputs=3, axis=1, splits=[4, 1, 51]),
        # Apply NMS to select best boxes. iou and score values match
        # https://github.com/ultralytics/ultralytics/blob/e7bd159a44cf7426c0f33ed9b413ef4439505a03/ultralytics/models/yolo/pose/predict.py#L34-L35
        # thresholds are arbitrarily chosen. adjust as needed
        SelectBestBoundingBoxesByNMS(iou_threshold=iou_threshold, score_threshold=score_threshold, has_mask_data=True),
        # Scale boxes and key point coords back to original image. Mask data has 17 key points per box.
        (ScaleNMSBoundingBoxesAndKeyPoints(num_key_points=17, layout='CHW'),
         [
             # A default connection from SelectBestBoundingBoxesByNMS for input 0
             # A connection from original image to input 1
             # A connection from the resized image to input 2
             # A connection from the LetterBoxed image to input 3
             # We use the three images to calculate the scale factor and offset.
             # With scale and offset, we can scale the bounding box and key points back to the original image.
             utils.IoMapEntry("BGRImageCHW", producer_idx=0, consumer_idx=1),
             utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
             utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
        ]),
    ]

    pipeline.add_post_processing(post_processing_steps)

    print("Updating model ...")

    new_model = pipeline.run(model)
    print("Pre/post proceessing added.")

    # run shape inferencing to validate the new model. shape inferencing will fail if any of the new node
    # types or shapes are incorrect. infer_shapes returns a copy of the model with ValueInfo populated,
    # but we ignore that and save new_model as it is smaller due to not containing the inferred shape information.
    _ = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    onnx.save_model(new_model, str(onnx_model_path.resolve()))
    print("Updated model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model with pre/post processing')
    parser.add_argument('--pt_model_path', type=Path, required=True, help='Path to the PyTorch model')
    parser.add_argument('--onnx_model_path', type=Path, required=True, help='Path to save the ONNX model')
    parser.add_argument('--model_size', type=int, default=640, help='Size of the model input (default: 640)')
    parser.add_argument('--iou_threshold', type=float, default=0.7, help='IoU threshold for NMS (default: 0.7)')
    parser.add_argument('--score_threshold', type=float, default=0.25, help='Score threshold for NMS (default: 0.25)')
    args = parser.parse_args()

    generate_model(
        args.pt_model_path,
        args.onnx_model_path,
        args.model_size,
        args.iou_threshold,
        args.score_threshold,
    )

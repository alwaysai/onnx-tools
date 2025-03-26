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
    Unsqueeze,
    Squeeze,
    Transpose,
    Split,
    SelectBestBoundingBoxesByNMS,
    ScaleNMSBoundingBoxesAndKeyPoints,
    utils,
)
from pathlib import Path
import ultralytics


def generate_model(pt_model_path: Path, onnx_model_path: Path):
    # Load the PyTorch model
    model = ultralytics.YOLO(str(pt_model_path))

    # Export to ONNX
    success = model.export(format="onnx", opset=18)
    assert success, "Failed to export model to ONNX"

    # Load the ONNX model
    model = onnx.load(str(onnx_model_path.resolve(strict=True)))

    # Add pre-processing and post-processing steps
    onnx_opset = 18
    # create a ValueInfoProto object in ONNX.  A ValueInfoProto describes a tensor, including its name,
    # data type, and shape.
    inputs = [create_named_value("image_bytes", onnx.TensorProto.UINT8, ["num_bytes"])]
    pipeline = PrePostProcessor(inputs, onnx_opset)

    pre_processing_steps = [
        ConvertImageToBGR(name="BGRImageHWC"),  # jpg/png image to BGR in HWC layout
        ChannelsLastToChannelsFirst(name="BGRImageCHW"),  # HWC to CHW
        # Resize to match model input. Uses not_larger as we use LetterBox to pad as needed.
        Resize((640, 640), policy='not_larger', layout='CHW'),
        LetterBox(target_shape=(640, 640), layout='CHW'),  # padding or cropping the image to (640, 640)
        ImageBytesToFloat(),  # Convert to float in range 0..1
        Unsqueeze([0]),  # add batch, CHW --> 1CHW
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
        SelectBestBoundingBoxesByNMS(iou_threshold=0.7, score_threshold=0.25, has_mask_data=True),
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
    args = parser.parse_args()

    generate_model(args.pt_model_path, args.onnx_model_path)

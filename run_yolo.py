import numpy as np
import cv2
import urllib.request
import os
from PIL import Image
from io import BytesIO
import onnxruntime as ort

# -----------------------------------------------------------
# 1) Define constants: COCO class names, model/image URLs, etc.
# -----------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_URL = "http://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
MODEL_PATH = "yolox_nano.onnx"

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
IMAGE_PATH = "backup_cat_416x416.jpg"  # A local file name to cache the image.

# The model is trained around a default input size of 416x416 (for YOLOX Nano).
INPUT_WIDTH  = 416
INPUT_HEIGHT = 416


# -----------------------------------------------------------
# 2) Utility function: download the model/image if not cached
# -----------------------------------------------------------
def ensure_file_downloaded(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"Found cached file: {file_path}")


# -----------------------------------------------------------
# 3) The YOLOX decoding logic
# -----------------------------------------------------------
def decode_outputs(outputs, input_size, p6=False):
    """
    YOLOX produces raw outputs in shape [N, 85], where:
      - The first 4 elements are [tx, ty, tw, th] in a 'grid+stride' space.
      - The 5th element is object confidence.
      - The last 80 elements are class probabilities.
    We must decode tx, ty, tw, th using the feature map grid and stride.

    Args:
        outputs: raw model outputs, shape [N, 85].
        input_size: (height, width) of model input
        p6: whether the model includes stride=64
    Returns:
        Decoded bounding boxes in shape [N, 4],
        plus the object confidence, plus class probabilities,
        still in the same 1:1 indexing as outputs.
    """
    # Strides for YOLOX Nano typically are [8, 16, 32]. If p6 is True, then we also have stride 64.
    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]

    # We'll figure out how many elements correspond to each stride:
    height, width = input_size
    hs = [height // s for s in strides]
    ws = [width  // s for s in strides]

    # Build up a single grid (x,y) plus expanded strides
    grids = []
    expanded_strides = []

    start = 0
    for hsize, wsize, stride in zip(hs, ws, strides):
        length = int(hsize * wsize)
        # make a grid for this stride
        grid_y, grid_x = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing='ij')
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        # build strides
        shape = grid.shape[:2]  # (1, hsize*wsize)
        expanded_strides.append(np.full((*shape, 1), stride))
        start += length

    grids = np.concatenate(grids, axis=1)             # shape: (1, sum of all grids, 2)
    expanded_strides = np.concatenate(expanded_strides, axis=1)  # shape: (1, sum of all grids, 1)

    # Now, the raw outputs come in shape (N, 85). We'll do them as if we have [1, N, 85] for matching dimensions:
    outputs = np.expand_dims(outputs, axis=0)

    # transform x,y
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    # transform w,h
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return np.squeeze(outputs, axis=0)


def xywh2xyxy(box_xywh):
    """
    Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    """
    box_xyxy = np.zeros_like(box_xywh)
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2.0  # x1
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2.0  # y1
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2.0  # x2
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2.0  # y2
    return box_xyxy


# -----------------------------------------------------------
# 4) Basic NMS to filter predictions
# -----------------------------------------------------------
def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    A basic NMS (Non-Maximum Suppression).
    Args:
        boxes: shape [N, 4], each row is [x1, y1, x2, y2]
        scores: shape [N], confidence for each box
        iou_threshold: threshold for deciding overlaps
    Returns:
        indices of boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def main():
    # -----------------------------------------------------------
    # A) Download/copy model & image locally if not present
    # -----------------------------------------------------------
    ensure_file_downloaded(MODEL_URL, MODEL_PATH)
    ensure_file_downloaded(IMAGE_URL, IMAGE_PATH)

    # -----------------------------------------------------------
    # B) Load the image and prepare input
    # -----------------------------------------------------------
    original_image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    orig_h, orig_w = original_image.shape[:2]

    # Resize/normalize for YOLOX
    input_image = cv2.resize(original_image, (INPUT_WIDTH, INPUT_HEIGHT))
    input_image = input_image.astype(np.float32) / 255.0
    # NCHW
    input_image = np.transpose(input_image, (2, 0, 1))[None, ...]

    # -----------------------------------------------------------
    # C) Initialize onnxruntime session & run inference
    # -----------------------------------------------------------
    session = ort.InferenceSession(MODEL_PATH)
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    # YOLOX output shape will be [3549, 85] for (416x416) with strides [8,16,32]

    raw_preds = outputs[0]  # shape: (3549, 85)
    raw_preds = raw_preds.squeeze(0)  # now shape = (3549, 85)

    # -----------------------------------------------------------
    # D) Decode the raw predictions to XYXY boxes
    # -----------------------------------------------------------

    decoded_preds = decode_outputs(raw_preds, (INPUT_HEIGHT, INPUT_WIDTH), p6=False)
    # decode_outputs returns shape [3549, 85], but now the first 4 columns are [cx, cy, w, h] in pixel coords

    # Split the columns
    boxes_xywh = decoded_preds[:, 0:4]  # [cx, cy, w, h]
    obj_conf    = decoded_preds[:, 4]   # object confidence
    class_probs = decoded_preds[:, 5:]  # per-class probabilities

    print("boxes_xywh.shape =", boxes_xywh.shape)  # (3549, 4)
    print("obj_conf.shape =", obj_conf.shape)  # (3549,)
    print("class_probs.shape =", class_probs.shape)  # (3549, 80)

    # Convert to [x1,y1,x2,y2]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # -----------------------------------------------------------
    # E) For each candidate: final score = obj_conf * class_conf
    # -----------------------------------------------------------
    final_scores = []
    final_class_ids = []

    for i in range(len(boxes_xyxy)):
        class_id = np.argmax(class_probs[i])
        class_conf = class_probs[i][class_id]
        score = obj_conf[i] * class_conf
        final_scores.append(score)
        final_class_ids.append(class_id)

    final_scores = np.array(final_scores)
    final_class_ids = np.array(final_class_ids)

    # Filter by some confidence threshold
    conf_thres = 0.3
    mask = final_scores >= conf_thres

    boxes_xyxy = boxes_xyxy[mask]
    scores = final_scores[mask]
    class_ids = final_class_ids[mask]

    if len(scores) == 0:
        print("No predictions above confidence threshold.")
        return

    # -----------------------------------------------------------
    # F) Perform Non-Maximum Suppression
    # -----------------------------------------------------------
    keep_indices = non_max_suppression(boxes_xyxy, scores, iou_threshold=0.45)
    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]

    # -----------------------------------------------------------
    # G) Print top detection (highest confidence)
    #    Note: The boxes are in the resized 416x416 space. You can
    #    scale them back to the original image if needed.
    # -----------------------------------------------------------
    max_id = np.argmax(scores)
    box = boxes_xyxy[max_id]
    cat = COCO_CLASSES[class_ids[max_id]]
    sc = scores[max_id]

    # Scale back to original image coordinates if you want:
    # scale_x = orig_w / float(INPUT_WIDTH)
    # scale_y = orig_h / float(INPUT_HEIGHT)
    # x1, y1, x2, y2 = box
    # x1 *= scale_x
    # x2 *= scale_x
    # y1 *= scale_y
    # y2 *= scale_y

    print("Top prediction:")
    print(f" Class:       {cat}")
    print(f" Confidence:  {sc:.3f}")
    print(f" BBox (x1,y1,x2,y2) in 416x416 space: {box}")


if __name__ == "__main__":
    main()
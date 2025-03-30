import os
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# ----------------------------------------------------------------------------------------
# 1) Constants and COCO class list
# ----------------------------------------------------------------------------------------

# Official Ultralytics YOLOv3 ONNX model (80-class, COCO-trained)
# Source: https://github.com/ultralytics/yolov3/releases (v9.6.0)
YOLOV3_ONNX_URL = "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.onnx"
YOLOV3_ONNX_PATH = "tiny-yolov3-11.onnx"

# Example cat image from COCO val2017
CAT_IMAGE_URL  = "http://images.cocodataset.org/val2017/000000039769.jpg"
CAT_IMAGE_PATH = "iStock-671533798.jpg"

# YOLOv3 expects 416x416 by default (this ONNX is usually 416-based)
INPUT_WIDTH  = 416
INPUT_HEIGHT = 416

# Confidence, IoU thresholds
CONF_THRES = 0.3
IOU_THRES  = 0.45

# Standard 80-class COCO names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Standard YOLOv3 anchor boxes, grouped by the 3 output layers (from largest stride to smallest)
# Each sub-list has 3 anchors, each anchor is (width, height)
# They correspond to strides [32, 16, 8] in that order.
YOLOV3_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # for scale 32
    [(30, 61),  (62, 45),   (59, 119)],   # for scale 16
    [(10, 13),  (16, 30),   (33, 23)],    # for scale 8
]

# The 3 detection layers produce feature maps at sizes:
#   layer 0: (416/32) = 13x13
#   layer 1: (416/16) = 26x26
#   layer 2: (416/8)  = 52x52
# or smaller if a different input size is used.


# ----------------------------------------------------------------------------------------
# 2) Utility: Download if missing
# ----------------------------------------------------------------------------------------
def ensure_file_downloaded(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"Found cached file: {filepath}")


# ----------------------------------------------------------------------------------------
# 3) Sigmoid + box decode functions
# ----------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bbox_iou(box1, box2):
    """Compute IoU of two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Basic NMS. Returns indices of boxes to keep."""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = np.array([bbox_iou(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]
    return keep


# ----------------------------------------------------------------------------------------
# 4) Parse YOLOv3 layer output
# ----------------------------------------------------------------------------------------
def parse_yolov3_layer(layer_output, anchors, stride, conf_threshold):
    """
    layer_output: shape (1, 3*(5+80), H, W) = (1, 255, H, W).
      After flattening to (3, 85, H, W),
        5+80 means: [tx, ty, tw, th, obj_conf, class_scores...]
    anchors: list of 3 (anchor_w, anchor_h)
    stride: the stride for this layer (32, 16, or 8)
    Returns list of (x1, y1, x2, y2, conf, class_id)
    """
    _, num_channels, hsize, wsize = layer_output.shape
    assert num_channels == 3 * (5 + len(COCO_CLASSES))  # 3 anchors, 5+80
    # Reshape to [3, 85, h, w]
    layer_output = layer_output.reshape(3, 5 + len(COCO_CLASSES), hsize, wsize)

    boxes = []
    # For each of the 3 anchors
    for a_i, (anchor_w, anchor_h) in enumerate(anchors):
        # Extract the sub-tensor for this anchor
        # shape: (5+80, h, w)
        data = layer_output[a_i]

        # Each item in data has shape (h, w). We'll parse them:
        # data[0] = tx, data[1] = ty, data[2] = tw, data[3] = th,
        # data[4] = obj_conf, data[5..] = class scores
        tx = data[0]
        ty = data[1]
        tw = data[2]
        th = data[3]
        obj_conf = data[4]

        class_score_map = data[5:]  # shape (80, h, w)

        # Sigmoid for x, y, objectness, class scores
        # (some exports might have partial sigmoid done, but standard YOLOv3 requires it)
        tx = sigmoid(tx)
        ty = sigmoid(ty)
        obj_conf = sigmoid(obj_conf)
        class_score_map = sigmoid(class_score_map)

        # Then for w, h we do "exp(tw)*anchor_w, exp(th)*anchor_h"
        tw = np.exp(tw) * anchor_w
        th = np.exp(th) * anchor_h

        # Then convert from center x,y to absolute coords
        # Each grid cell is "stride" in pixel space
        # Grid cell offsets:
        grid_x = np.tile(np.arange(wsize), (hsize, 1))
        grid_y = np.tile(np.arange(hsize).reshape(-1,1), (1, wsize))

        # x,y in pixel coords
        cx = (tx + grid_x) * stride
        cy = (ty + grid_y) * stride

        # w,h in pixel coords
        w_box = tw * stride
        h_box = th * stride

        # Now for each cell we can compute final x1,y1,x2,y2
        # shape: (h, w)
        x1 = cx - w_box / 2.0
        y1 = cy - h_box / 2.0
        x2 = cx + w_box / 2.0
        y2 = cy + h_box / 2.0

        # Flatten them
        x1 = x1.flatten()
        y1 = y1.flatten()
        x2 = x2.flatten()
        y2 = y2.flatten()

        obj_conf = obj_conf.flatten()
        class_score_map = class_score_map.reshape(len(COCO_CLASSES), -1)  # shape: (80, h*w)

        # For each cell we find the best class
        class_ids = np.argmax(class_score_map, axis=0)
        class_scores = class_score_map.max(axis=0)

        # final_conf = obj_conf * class_scores
        final_conf = obj_conf * class_scores

        # Filter by threshold
        keep_mask = final_conf >= conf_threshold
        if np.sum(keep_mask) == 0:
            continue

        # Gather the kept indices
        x1 = x1[keep_mask]
        y1 = y1[keep_mask]
        x2 = x2[keep_mask]
        y2 = y2[keep_mask]
        final_conf = final_conf[keep_mask]
        cids = class_ids[keep_mask]

        # Collect boxes
        for j in range(len(x1)):
            boxes.append((x1[j], y1[j], x2[j], y2[j], final_conf[j], cids[j]))

    return boxes


# ----------------------------------------------------------------------------------------
# 5) Full detection flow for YOLOv3 (one image)
# ----------------------------------------------------------------------------------------
def detect_image_with_yolov3(session, image_path):
    """
    1. Loads the image & resizes to 416x416
    2. Runs the ONNX model => 3 output layers
    3. parse each layer -> bounding boxes
    4. NMS
    5. Return highest-confidence 'cat' bounding box if found, else None
    """
    # A) Read image, store original size
    original_img = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = original_img.shape[:2]

    # B) Preprocess to 416x416
    img = cv2.resize(original_img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img[:, :, ::-1]  # BGR->RGB if needed (some exports want BGR, but it's often okay)
    img = img.astype(np.float32) / 255.0
    # NCHW
    img = np.transpose(img, (2, 0, 1))[None, ...]

    # C) Forward pass
    #   The official Ultralytics YOLOv3 ONNX has 3 outputs => layer0, layer1, layer2
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    # Expect outputs[0].shape -> (1,255,13,13)  (stride=32)
    #        outputs[1].shape -> (1,255,26,26)  (stride=16)
    #        outputs[2].shape -> (1,255,52,52)  (stride=8)

    # D) For each of the 3 outputs, parse using anchors & stride
    # strides = [32,16,8], matching YOLOV3_ANCHORS order
    strides = [32, 16, 8]
    all_boxes = []
    for layer_i, (layer_output, anchors, stride) in enumerate(zip(outputs, YOLOV3_ANCHORS, strides)):
        # parse that layer
        boxes = parse_yolov3_layer(layer_output, anchors, stride, CONF_THRES)
        all_boxes.extend(boxes)  # each item: (x1,y1,x2,y2,conf,class_id)

    if not all_boxes:
        return None

    # E) Convert to Numpy for NMS
    all_boxes_np = np.array([b[:4] for b in all_boxes], dtype=np.float32)  # (N,4)
    all_scores_np = np.array([b[4] for b in all_boxes], dtype=np.float32)  # (N,)
    all_cids_np   = np.array([b[5] for b in all_boxes], dtype=np.int32)    # (N,)

    # F) NMS
    keep_idx = non_max_suppression(all_boxes_np, all_scores_np, IOU_THRES)
    kept_boxes  = all_boxes_np[keep_idx]
    kept_scores = all_scores_np[keep_idx]
    kept_cids   = all_cids_np[keep_idx]

    # G) Look specifically for "cat"
    cat_indices = [i for i, cid in enumerate(kept_cids) if COCO_CLASSES[cid] == "cat"]
    if not cat_indices:
        return None

    # pick best cat
    best_idx = max(cat_indices, key=lambda i: kept_scores[i])
    cat_conf = kept_scores[best_idx]
    x1, y1, x2, y2 = kept_boxes[best_idx]

    # H) scale coords back to original
    scale_x = orig_w / float(INPUT_WIDTH)
    scale_y = orig_h / float(INPUT_HEIGHT)
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    # return bounding box + confidence
    return (x1, y1, x2, y2, cat_conf)


# ----------------------------------------------------------------------------------------
# 6) Main script
# ----------------------------------------------------------------------------------------
def main():
    # 1) Download ONNX model + cat image if not present
    ensure_file_downloaded(YOLOV3_ONNX_URL, YOLOV3_ONNX_PATH)
    ensure_file_downloaded(CAT_IMAGE_URL, CAT_IMAGE_PATH)

    # 2) Load model with onnxruntime
    session = ort.InferenceSession(YOLOV3_ONNX_PATH)
    print("Loaded YOLOv3 ONNX model successfully.")

    # 3) Detect
    cat_result = detect_image_with_yolov3(session, CAT_IMAGE_PATH)
    if cat_result is None:
        print("No cat found in the image!")
    else:
        x1, y1, x2, y2, conf = cat_result
        print("CAT DETECTED!")
        print(f"  Confidence: {conf:.2f}")
        print(f"  Bounding Box [x1,y1,x2,y2]: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


if __name__ == "__main__":
    main()
import os
import sys
import time
import urllib.request
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

# ---------------------------------------------------
# 1) COCO labels and constants
# ---------------------------------------------------
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

MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
    "yolox_tiny.onnx"
)
MODEL_PATH = "yolox_tiny.onnx"

INPUT_WIDTH, INPUT_HEIGHT = 416, 416
CONF_THRESHOLD = 0.3
IOU_THRESHOLD  = 0.45


# ----------------------------------------------------------------
# 2) Utilities
# ----------------------------------------------------------------
def ensure_file_downloaded(url, file_path):
    """Download file from `url` to `file_path` if it's not there."""
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"Found cached file: {file_path}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(box_xywh):
    """
    Convert [cx, cy, w, h] -> [x1, y1, x2, y2].
    """
    out = np.zeros_like(box_xywh)
    out[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2.0  # x1
    out[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2.0  # y1
    out[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2.0  # x2
    out[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2.0  # y2
    return out

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    Basic NMS. Returns indices of boxes to keep.
    boxes: (N, 4) -> x1,y1,x2,y2
    scores: (N,)
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

def decode_outputs(outputs, input_size, p6=False):
    """
    YOLOX decoding:
    - Sigmoid x,y,obj_conf,class_probs
    - exp w,h
    - Scale/shift by grid & stride
    """
    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]
    height, width = input_size
    hs = [height // s for s in strides]
    ws = [width  // s for s in strides]

    grids = []
    expanded_strides = []
    for hsize, wsize, stride in zip(hs, ws, strides):
        grid_y, grid_x = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing='ij')
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)

    outputs = np.expand_dims(outputs, 0)

    # 1) x,y -> sigmoid
    outputs[..., 0:2] = sigmoid(outputs[..., 0:2])
    # 2) w,h -> exp
    outputs[..., 2:4] = np.exp(outputs[..., 2:4])
    # 3) object_conf + class_probs -> sigmoid
    outputs[..., 4:] = sigmoid(outputs[..., 4:])

    # apply scale + offset
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = outputs[..., 2:4] * expanded_strides

    return np.squeeze(outputs, axis=0)

# ----------------------------------------------------------------
# 3) The detection function
# ----------------------------------------------------------------
def detect_cat_in_image(session, image_path):
    """
    Runs YOLOX Nano on `image_path`. Returns a tuple (best_box, best_score) for 'cat'
    if found, otherwise None.
    best_box is in original image coords as [x1,y1,x2,y2].
    """
    original_image = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = original_image.shape[:2]

    # Preprocess
    inp = cv2.resize(original_image, (INPUT_WIDTH, INPUT_HEIGHT))
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    # Inference
    raw = session.run(None, {session.get_inputs()[0].name: inp})[0]
    raw = raw.squeeze(0)  # shape: (3549, 85)

    # Decode
    decoded = decode_outputs(raw, (INPUT_HEIGHT, INPUT_WIDTH), p6=False)  # (3549, 85)
    box_xywh   = decoded[:, :4]
    obj_conf   = decoded[:, 4]
    class_probs = decoded[:, 5:]

    # Convert [cx,cy,w,h] -> [x1,y1,x2,y2]
    boxes_xyxy = xywh2xyxy(box_xywh)

    # Final scores
    final_scores = []
    final_class_ids = []
    for i in range(len(boxes_xyxy)):
        cid = np.argmax(class_probs[i])
        class_conf = class_probs[i][cid]
        score = obj_conf[i] * class_conf
        final_scores.append(score)
        final_class_ids.append(cid)

    final_scores = np.array(final_scores)
    final_class_ids = np.array(final_class_ids)

    # Filter
    mask = final_scores >= CONF_THRESHOLD
    boxes_xyxy = boxes_xyxy[mask]
    scores     = final_scores[mask]
    cids       = final_class_ids[mask]

    if len(scores) == 0:
        return None

    # NMS
    keep = non_max_suppression(boxes_xyxy, scores, iou_threshold=IOU_THRESHOLD)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    cids   = cids[keep]

    # Check if any are 'cat'
    cat_indices = [i for i, cid in enumerate(cids) if COCO_CLASSES[cid] == 'cat']
    if not cat_indices:
        return None

    # Get best cat
    best_idx = max(cat_indices, key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_box_416 = boxes_xyxy[best_idx]  # in resized space

    # Scale back to original coords
    scale_x = orig_w / float(INPUT_WIDTH)
    scale_y = orig_h / float(INPUT_HEIGHT)
    x1, y1, x2, y2 = best_box_416
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    return [x1, y1, x2, y2], best_score


# ----------------------------------------------------------------
# 4) Main function: iterate val2017 until cat is found
# ----------------------------------------------------------------
def main():
    # 1) Download model if needed, load session
    ensure_file_downloaded(MODEL_URL, MODEL_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    # 2) Check images in val2017
    val_dir = "val2017"
    files = sorted(os.listdir(val_dir))
    image_paths = [os.path.join(val_dir, f) for f in files
                   if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    total = len(image_paths)
    if total == 0:
        print(f"No images found in {val_dir}")
        return

    # 3) Loop and detect cat
    print(f"Scanning {total} images in {val_dir} for a cat (CTRL+C to stop)...")
    for i, path in enumerate(image_paths, start=1):
        # Update status line
        sys.stdout.write(f"\rProcessed {i}/{total} images...")
        sys.stdout.flush()

        result = detect_cat_in_image(session, path)
        if result is not None:
            box, score = result
            sys.stdout.write("\n")  # newline after progress
            sys.stdout.flush()
            print("Found a cat!")
            print(f" Image path: {path}")
            print(f" Confidence: {score:.2f}")
            print(f" Bounding box [x1, y1, x2, y2]: {box}")
            # return

        # Delay 0.1s to visually see the updates
        time.sleep(0.1)

    # If we finish the loop, no cat was found
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("No cat found in any image of val2017.")


if __name__ == "__main__":
    main()
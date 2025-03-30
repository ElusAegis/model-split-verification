import numpy as np
import cv2
import json
import urllib.request
import os
from PIL import Image
from io import BytesIO
import onnxruntime as ort

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

def main():
    model_url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
    model_path = "yolox_nano.onnx"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = urllib.request.urlopen(image_url)
    original_image = np.array(Image.open(BytesIO(response.read())).convert('RGB'))

    input_image = cv2.resize(original_image, (416, 416))
    img = input_image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    # Save image to disk as input.json, where {input_data: [image.flatten().tolist()]}
    with open("input.json", "w") as f:
        json.dump({"input_data": [img.flatten().tolist()]}, f)

    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {session.get_inputs()[0].name: img})

    predictions = outputs[0][0]  # [3549, 85]

    confidence_threshold = 0.1
    input_w, input_h = 416, 416

    boxes = []
    for pred in predictions:
        x, y, w, h, obj_conf = pred[:5]
        class_probs = pred[5:]
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]

        score = obj_conf * class_conf
        if score >= confidence_threshold:
            box = {
                "x": (x - w / 2) * input_w,
                "y": (y - h / 2) * input_h,
                "width": w * input_w,
                "height": h * input_h,
                "class_id": class_id,
                "class_name": COCO_CLASSES[class_id],
                "score": score
            }
            boxes.append(box)

    if boxes:
        top_box = max(boxes, key=lambda x: x["score"])
        print(f"Top prediction:\n Class: {top_box['class_name']}\n Bounding Box: {top_box}")
    else:
        print("No predictions above confidence threshold.")


if __name__ == "__main__":
    main()
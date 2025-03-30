import torch
import torch.nn as nn
import torch.nn.functional as F


class CatRecognitionModel(nn.Module):
    """
    A toy CNN model that:
      1) Slices a region from the full image using (x1, y1, x2, y2).
      2) Resizes it to T×T.
      3) Classifies whether it's a cat or not (binary classification).
    """

    def __init__(self, T=28):
        super(CatRecognitionModel, self).__init__()
        self.T = T
        # A minimal CNN for demonstration:
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # We’ll output 2 logits: [logit_not_cat, logit_cat]
        self.fc = nn.Linear(16 * T * T, 2)

    def forward(self, image, coords):
        """
        image: [3, H, W]  (e.g., [channels=3, height=100, width=100])
        coords: [4]      (x1, y1, x2, y2)

        Output: [2] logits
        """
        x1, y1, x2, y2 = coords
        # Slice the region
        sub_img = image[:, y1:y2, x1:x2].unsqueeze(0)  # shape: [1, 3, subH, subW]

        # Downsample/resize to T×T
        sub_img_resized = F.interpolate(sub_img, size=(self.T, self.T), mode='bilinear')

        # Run through the CNN
        x = F.relu(self.conv1(sub_img_resized))
        x = F.relu(self.conv2(x))

        # Flatten for the fully-connected layer
        x = x.view(x.size(0), -1)  # shape: [1, 16*T*T]

        out = self.fc(x)  # shape: [1, 2]
        return out[0]   # return as [2]


def test_model():
    # 1) Create a random input image: shape [3, 100, 100]
    #    In practice, you'd load real image data.
    image = torch.randn(3, 100, 100)

    # 2) Define coords for the sub-region
    #    For example, let's pick a rectangle:
    coords = torch.tensor([10, 20, 40, 60], dtype=torch.int64)  # shape: [4]

    # 3) Instantiate our model
    T = 28  # Desired sub-chunk size
    model = CatRecognitionModel(T=T)

    # 4) Run a forward pass
    logits = model(image, coords)
    print("Logits:", logits)  # [2]
    probs = nn.Softmax(dim=0)(logits)  # since there's no batch dimension
    print("Cat probability:", probs[1].item())

    # 5) Export to ONNX
    onnx_filename = "cat_recognition_model.onnx"
    torch.onnx.export(
        model,                # model to be exported
        (image, coords),      # example inputs (tuple)
        onnx_filename,
        input_names=["image", "coords"],
        output_names=["logits"],
        opset_version=13      # or whichever opset version you require
    )
    print(f"Exported to {onnx_filename}")

    # 6) Save sample input to input.json
    import json
    sample_input = {
        "image": image.numpy().tolist(),
        "coords": coords.numpy().tolist()
    }
    with open("input.json", "w") as f:
        json.dump(sample_input, f, indent=4)
    print("Sample input saved to input.json")

    # 7) Run the exported ONNX model using onnxruntime and verify the output
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime is not installed. Please install it to run the ONNX model.")
        return

    import numpy as np
    sess = onnxruntime.InferenceSession(onnx_filename)
    image_np = image.numpy().astype(np.float32)  # shape: (3, 100, 100)
    coords_np = coords.numpy().astype(np.int64)
    inputs = {"image": image_np, "coords": coords_np}
    onnx_output = sess.run(None, inputs)
    print("ONNX model output:", onnx_output)


if __name__ == "__main__":
    test_model()
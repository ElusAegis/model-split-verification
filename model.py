import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

class SimpleCNN(nn.Module):
    """
    A small CNN for demonstration:
      1) Expects an input image of shape [C, H, W].
      2) Adds a batch dimension internally.
      3) Applies a Conv2d -> ReLU -> Flatten -> Linear pipeline.
      4) Produces a small output vector of shape [10].
    """
    def __init__(self, in_channels=3, out_dim=10):
        super(SimpleCNN, self).__init__()
        # A single 2D conv layer for demonstration
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        # A fully connected layer to produce an output of size out_dim (e.g. 10)
        self.fc = nn.Linear(8 * 28 * 28, out_dim)

    def forward(self, x):
        # x has shape [C, H, W]. We add batch dimension: [1, C, H, W].
        x = x.unsqueeze(0)

        # 1) Convolution
        x = self.conv(x)       # shape: [1, 8, H, W]
        x = F.relu(x)

        # 2) Flatten
        x = x.view(x.size(0), -1)  # shape: [1, 8*H*W]

        # 3) Linear
        x = self.fc(x)             # shape: [1, out_dim]
        return x[0]                # return shape: [out_dim]


def main():
    # 1) Instantiate the model
    model = SimpleCNN(in_channels=3, out_dim=10)

    # 2) Create sample input: shape [3, 28, 28]
    #    (3 channels, 28x28 image).
    sample_input = torch.randn(3, 28, 28)

    # 3) Forward pass
    output = model(sample_input)
    print("Model output:\n", output)  # shape: [10]

    # 4) Export to ONNX
    onnx_filename = "simple_cnn.onnx"
    torch.onnx.export(
        model,
        (sample_input,),
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=13
    )
    print(f"Exported to {onnx_filename}")

    # 5) Create and save JSON input data (flatten the input)
    flattened = sample_input.reshape(-1).detach().cpu().numpy().tolist()
    data = {
        "input_data": flattened
    }
    with open("input.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Sample input saved to input.json")

    # 6) (Optional) Run a quick check in onnxruntime
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime is not installed. Skipping runtime test.")
        return

    sess = onnxruntime.InferenceSession(onnx_filename)
    arr = np.array(flattened, dtype=np.float32).reshape(sample_input.shape)
    onnx_out = sess.run(None, {"input": arr})
    print("ONNX model output:\n", onnx_out)


if __name__ == "__main__":
    main()
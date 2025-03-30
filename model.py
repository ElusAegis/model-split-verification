import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import subprocess
import sys


class SimpleCNN(nn.Module):
    """
    A small CNN for demonstration:
      1) Expects an input image of shape [C, H, W].
      2) Adds a batch dimension internally.
      3) Applies a Conv2d -> ReLU -> Flatten -> Linear pipeline -> 2 logits.
      4) Produces just the cat probability (softmaxed).
    """
    def __init__(self, in_channels=3):
        super(SimpleCNN, self).__init__()
        # A single 2D conv layer for demonstration
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        # We'll produce 2 logits: [logit_not_cat, logit_cat]
        self.fc = nn.Linear(8 * 28 * 28, 2)

    def forward(self, x):
        # x has shape [C, H, W]. We add batch dimension: [1, C, H, W].
        x = x.unsqueeze(0)

        # 1) Convolution + ReLU
        x = self.conv(x)  # shape: [1, 8, H, W]
        x = F.relu(x)

        # 2) Flatten
        x = x.view(x.size(0), -1)  # shape: [1, 8*H*W]

        # 3) Compute logits
        logits = self.fc(x)        # shape: [1, 2]

        # 4) Convert logits -> probabilities, shape: [1, 2]
        probs = F.softmax(logits, dim=1)

        # Return just the cat probability
        cat_prob = probs[:, 1]     # shape: [1]
        return cat_prob[0]         # shape: ()

def run_command(command: str):
    """
    Runs the command via subprocess. If it fails, prints error and exits.
    """
    print(f"Running: {command}")
    try:
        subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError:
        print("Command failed, stopping.")
        sys.exit(1)


def main():
    # 1) Instantiate the model
    model = SimpleCNN(in_channels=3)

    # 2) Create sample input: shape [3, 28, 28]
    sample_input = torch.randn(3, 28, 28)

    # 3) Forward pass
    output = model(sample_input)
    print("Cat probability:", output.item())

    # 4) Export to ONNX
    onnx_filename = "cat_model.onnx"
    torch.onnx.export(
        model,
        (sample_input,),
        onnx_filename,
        input_names=["input"],
        output_names=["cat_prob"],
        opset_version=13
    )
    print(f"Exported to {onnx_filename}")

    # 5) Create and save JSON input data
    # We flatten the input, but store it as a nested list: [[...values...]]
    flattened = sample_input.reshape(-1).detach().cpu().numpy().tolist()
    data = {
        "input_data": [flattened]
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

    # 7) Attempt to run all ezkl commands in sequence:
    commands = [
        f"ezkl gen-settings --model {onnx_filename}",
        f"ezkl calibrate-settings --model {onnx_filename} --data input.json",
        f"ezkl compile-circuit --model {onnx_filename}",
        "ezkl setup",
        "ezkl gen-witness --data input.json",
        "ezkl prove"
    ]

    for cmd in commands:
        run_command(cmd)
    print("All commands completed successfully.")


if __name__ == "__main__":
    main()
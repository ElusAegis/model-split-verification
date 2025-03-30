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
      1) Accepts an input of shape [C, H, W].
      2) Upsamples to the nearest multiples of 28 (via nearest-neighbor).
      3) Downsamples to exactly 28x28 (via AdaptiveAvgPool2d).
      4) Applies a Conv2d -> ReLU -> Flatten -> Linear pipeline -> 2 logits.
      5) Produces just the cat probability (softmaxed).
    """
    def __init__(self, in_channels=3):
        super(SimpleCNN, self).__init__()

        # Step 1: Downsample to 28x28
        self.down_pool = nn.AdaptiveAvgPool2d((28, 28))

        # Step 2: Convolution
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        # We'll produce 2 logits: [logit_not_cat, logit_cat]
        self.fc = nn.Linear(8 * 28 * 28, 2)

    def forward(self, x):
        # x: [C, H, W]
        x = x.unsqueeze(0)  # shape: [1, C, H, W]

        # 1) Slice
        x = x[:, :, 30:86, 30:86]  # shape: [1, C, 56, 56]

        # 2) Downsample
        x = self.down_pool(x)   # shape: [1, C, 28, 28]

        # 3) Convolution + ReLU
        x = self.conv(x)       # shape: [1, 8, 28, 28]
        x = F.relu(x)

        # 4) Flatten
        x = x.view(x.size(0), -1)   # [1, 8*28*28]

        # 5) Compute logits
        logits = self.fc(x)         # [1, 2]
        probs = F.softmax(logits, dim=1)  # [1, 2]

        cat_prob = probs[:, 1]      # [1]
        return cat_prob[0]          # ()

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

    # 2) Create sample input of shape [3, 100, 50]
    sample_input = torch.randn(3, 160, 112)

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

    # 7) Attempt to run all ezkl commands in sequence, capturing their output:
    commands = [
        (f"ezkl gen-settings --model {onnx_filename}", "GEN SETTINGS"),
        (f"ezkl calibrate-settings --model {onnx_filename} --data input.json", "CALIBRATE SETTINGS"),
        (f"ezkl compile-circuit --model {onnx_filename}", "COMPILE CIRCUIT"),
        ("ezkl setup", "SETUP"),
        ("ezkl gen-witness --data input.json", "GENERATE WITNESS"),
        ("ezkl prove", "PROVE")
    ]

    for cmd, label in commands:
        try:
            result = subprocess.run(cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"✅ PASSED {label}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR IN {label}:\n{e.stdout}\n{e.stderr}")
            sys.exit(1)

    print("🎉 All commands completed successfully.")


if __name__ == "__main__":
    main()
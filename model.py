import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import subprocess
import sys
import time


def run_command(cmd, label):
    """
    Run a command in the shell, capturing its output.
    Returns a tuple of (stdout, stderr).
    """
    print(f"> RUNNING {label} ...", end="", flush=True)
    start_time = time.time()

    # Start the process
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Keep updating the elapsed time until the process ends
    while True:
        # Check if the process has finished
        returncode = process.poll()
        if returncode is not None:
            break

        elapsed = int(time.time() - start_time)
        # Overwrite the same line to avoid spamming
        sys.stdout.write(f"\r> RUNNING {label} ... ⏳{elapsed}s elapsed ⏳")
        sys.stdout.flush()
        time.sleep(0.005)

    # Final elapsed time
    total_time = (time.time() - start_time)

    # Process ended, gather output
    out, err = process.communicate()

    if returncode == 0:
        print(f"\r✅ PASSED {label} in {total_time:.2f}s")
    else:
        print(f"\r❌ ERROR IN {label} after {total_time:.2f}s:\n{out}\n{err}")
        sys.exit(1)


def build_row_selector(y1, y2, n):
    """
    Build a row-selection matrix R of shape ((y2 - y1) x n),
    picking rows [y1 .. (y2-1)] out of [0..n-1].

    y1, y2: ints (top and bottom, exclusive end)
    n: total height
    """
    row_count = y2 - y1  # 56 if y1=30, y2=86
    R = torch.zeros((row_count, n), dtype=torch.float32)
    for i in range(row_count):
        R[i, y1 + i] = 1.0
    return R


def build_col_selector(x1, x2, m):
    """
    Build a column-selection matrix C of shape (m x (x2 - x1)),
    picking columns [x1 .. (x2-1)] out of [0..m-1].

    x1, x2: ints (left and right, exclusive end)
    m: total width
    """
    col_count = x2 - x1  # 56 if x1=30, x2=86
    C = torch.zeros((m, col_count), dtype=torch.float32)
    for i in range(col_count):
        C[x1 + i, i] = 1.0
    return C


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

    def forward(self, x, R, C):
        channels, n, m = x.shape

        x_reshaped = x.view(channels, n, m)

        # Multiply: (sub_n, n) @ (n, m) => (sub_n, m)
        # then => (sub_n, m) @ (m, sub_m) => (sub_n, sub_m)
        sub_n = R.shape[0]
        sub_m = C.shape[1]

        # Matmul for each of the (batch_size*channels) slices
        #  => (batch_size*channels, sub_n, sub_m)
        # We'll do it in two steps for clarity:
        x_sub = torch.matmul(R, x_reshaped)  # shape (sub_n, m) for each slice
        x_sub = torch.matmul(x_sub, C)  # shape (sub_n, sub_m) for each slice

        # Reshape back to (batch_size, channels, sub_n, sub_m)
        x_sub = x_sub.view(1, channels, sub_n, sub_m)

        #
        #
        # # x: [C, H, W]
        # x = x.unsqueeze(0)  # shape: [1, C, H, W]
        #
        # # 1) Slice
        # x = x[:, :, 30:86, 30:86]  # shape: [1, C, 56, 56]

        # 2) Downsample
        x = self.down_pool(x_sub)  # shape: [1, C, 28, 28]

        # 3) Convolution + ReLU
        x = self.conv(x)  # shape: [1, 8, 28, 28]
        x = F.relu(x)

        # 4) Flatten
        x = x.view(x.size(0), -1)  # [1, 8*28*28]

        # 5) Compute logits
        logits = self.fc(x)  # [1, 2]
        probs = F.softmax(logits, dim=1)  # [1, 2]

        cat_prob = probs[:, 1]  # [1]
        return cat_prob[0]  # ()


def main():
    n, m = 64, 64
    x1, y1 = 5, 5
    x2, y2 = 61, 61

    # Build the selector matrices (crop 56x56)
    R = build_row_selector(y1, y2, n)  # shape [56, 200]
    C = build_col_selector(x1, x2, m)  # shape [200, 56]

    # 1) Instantiate the model
    model = SimpleCNN(in_channels=3)

    # 2) Create sample input of shape [3, 200, 200]
    sample_input = torch.randn(3, n, m)

    # 3) Forward pass (note we pass R, C now!)
    output = model(sample_input, R, C)  # <--- CHANGED
    print("🐱 Cat probability:", output.item())

    # 4) Export to ONNX
    onnx_filename = "cat_model.onnx"
    # NOTE: We now have 3 inputs: x, R, C
    # We'll name them "input_x", "input_R", "input_C" in the ONNX graph.
    torch.onnx.export(
        model,
        (sample_input, R, C),  # <--- CHANGED
        onnx_filename,
        input_names=["input_x", "input_R", "input_C"],  # <--- CHANGED
        output_names=["cat_prob"],
        opset_version=13
    )
    print(f"🗄️ Exported to {onnx_filename}")

    # 5) Create and save JSON input data
    # We'll store x, R, and C all flattened, so we can feed them to onnxruntime or ezkl.
    flattened_x = sample_input.reshape(-1).detach().cpu().numpy().tolist()
    flattened_R = R.reshape(-1).detach().cpu().numpy().tolist()
    flattened_C = C.reshape(-1).detach().cpu().numpy().tolist()

    data = {
        "input_data": [flattened_x, flattened_R, flattened_C],
        # "input_R": [],
        # "input_C": [flattened_C]
    }
    with open("input.json", "w") as f:
        json.dump(data, f, indent=4)
    print("🗄️ Sample input (x, R, C) saved to input.json")

    # 6) (Optional) Run a quick check in onnxruntime
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime is not installed. Skipping runtime test.")
        return

    # Rebuild the arrays from JSON-like structure
    arr_x = np.array(flattened_x, dtype=np.float32).reshape(sample_input.shape)
    arr_R = np.array(flattened_R, dtype=np.float32).reshape(R.shape)
    arr_C = np.array(flattened_C, dtype=np.float32).reshape(C.shape)

    sess = onnxruntime.InferenceSession(onnx_filename)
    # We feed the same 3 inputs to the model
    onnx_out = sess.run(
        None,
        {"input_x": arr_x, "input_R": arr_R, "input_C": arr_C}
    )
    print("🐱 ONNX model output:", onnx_out)

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
        run_command(cmd, label)

    print("🎉 All commands completed successfully.")


if __name__ == "__main__":
    main()

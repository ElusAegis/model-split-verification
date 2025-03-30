import torch
import torch.nn as nn
import json
import numpy as np

class MySimpleModel(nn.Module):
    def __init__(self):
        super(MySimpleModel, self).__init__()

    def forward(self, x):
        # Minimal operation: add 1 to every element
        return x + 1

def main():
    # 1) Instantiate the model
    model = MySimpleModel()

    # 2) Create sample input (e.g., shape [3, 2])
    sample_input = torch.randn(3, 2)

    # 3) Forward pass
    output = model(sample_input)
    print("Model output:\n", output)

    # 4) Export to ONNX (no ezkl usage)
    onnx_filename = "mysimplemodel.onnx"
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
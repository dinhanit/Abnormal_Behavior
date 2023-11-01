import torch
from param import *

# Load the model
model = torch.load("model/Weight")

# Create a random input tensor on the same device as the model
device = next(model.parameters()).device
x = torch.rand(1,1,171, device=device)

# Export the model to ONNX with specified input and output names
input_names = ["Featurex171"]  # Name for the input tensor
output_names = ["BinaryClassifier"]  # Name for the output tensor

# Export the model with input and output names
torch.onnx.export(model, x, "onnx/model.onnx", opset_version=11, input_names=input_names, output_names=output_names)

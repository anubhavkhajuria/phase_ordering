import torch
import torchvision.models as models
from torch_mlir import fx  
import torch_mlir

alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()

example_input = torch.randn(1, 3, 224, 224)  

mlir_module = fx.export_and_import(
    alex, 
    example_input, 
    output_type="linalg-on-tensors",
    func_name="alexnet"
)

with open("alexnet_linalg.mlir", "w") as f:
    f.write(str(mlir_module))

print("Wrote alexnet_linalg.mlir")

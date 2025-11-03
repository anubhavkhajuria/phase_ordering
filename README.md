# PyTorch to LLVM IR Pipeline

This repository provides pipelines for converting PyTorch models to LLVM IR and executing them. The system takes JPG images as input and performs predictions using the converted model.

## Repository Structure

- `model/` - Contains PyTorch model definitions and weights
- `llvm/` - LLVM IR generated code and compilation artifacts  
- `src/` - Source code for image processing and model conversion
- `examples/` - Sample images and usage examples
- `scripts/` - Helper scripts for conversion pipeline

## Dependencies

- PyTorch
- LLVM
- stb_image library for image loading
- CMake (>= 3.10)

## Building and Running

1. Clone the repository:
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

2. Build the project:
```bash
mkdir build && cd build
cmake ..
make
```

3. Run inference:
```bash
./infer_model path/to/image.jpg
```

## Input Format
- Supported image format: JPG
- Images are processed using stb_image library

## Notes
- Make sure input images match the model's expected dimensions
- Check logs in case of conversion/execution errors

## License
[MIT License](LICENSE)
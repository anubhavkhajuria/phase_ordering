# Phase Ordering in MLIR Compiler Optimizations

This repository demonstrates the **phase ordering problem** in compiler optimizations using MLIR (Multi-Level Intermediate Representation). It shows how different orderings of MLIR optimization passes can significantly impact the performance of deep learning models, specifically AlexNet for ImageNet classification.

## Table of Contents
- [Overview](#overview)
- [What is Phase Ordering?](#what-is-phase-ordering)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline Execution](#pipeline-execution)
  - [Compilation and Running](#compilation-and-running)
- [Understanding the Pipeline](#understanding-the-pipeline)
- [Experiments](#experiments)
- [Troubleshooting](#troubleshooting)
- [Results and Analysis](#results-and-analysis)

## Overview

This project explores how **the order of compiler optimization passes** affects the final performance of a neural network model. Using AlexNet as a case study, we convert PyTorch models to MLIR, apply different optimization pass orderings, lower to LLVM IR, and measure the impact on inference time and code quality.

**Key Question**: Does canonicalization → bufferization → loop-conversion produce better code than bufferization → canonicalization → loop-conversion?

## What is Phase Ordering?

The **phase ordering problem** is a fundamental challenge in compiler design where:
- Different sequences of the same optimization passes produce different results
- Some optimizations enable or block others
- The optimal ordering is program-dependent and non-obvious

**Example in this project:**
```
Order A: Canonicalize → CSE → Bufferize → Linalg-to-Loops → Lower-to-LLVM
Order B: Bufferize → Canonicalize → CSE → Linalg-to-Loops → Lower-to-LLVM
Order C: Linalg-to-Loops → Canonicalize → Bufferize → Lower-to-LLVM
```




## Prerequisites

### Required Tools

1. **LLVM/MLIR 22.0 or higher**
   ```bash
   # Check version
   mlir-opt --version
   # Should show: MLIR version 22.0 or higher
   ```

2. **torch-mlir**
   ```bash
   # Install torch-mlir
   pip install torch-mlir
   
   # Verify installation
   python -c "import torch_mlir; print(torch_mlir.__version__)"
   ```

3. **GCC Compiler (7.0 or higher)**
   ```bash
   gcc --version
   ```

4. **LLVM Static Compiler (llc)**
   ```bash
   llc --version
   ```

5. **Clang (for linking MLIR runtime)**
   ```bash
   clang --version
   ```

### Required Files
- `stb_image.h` - Single-header image loading library
- `stb_image_resize2.h` - Image resizing library
- `imagenet_classes.txt` - 1000 ImageNet class labels

### Installing LLVM/MLIR 22.0

```bash
# Option 1: Using official LLVM releases
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-22.0.0/clang+llvm-22.0.0-x86_64-linux-gnu-ubuntu-22.04.tar.xz
tar xf clang+llvm-22.0.0-x86_64-linux-gnu-ubuntu-22.04.tar.xz
export PATH=/path/to/clang+llvm-22.0.0/bin:$PATH

# Option 2: Build from source
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

ninja install

export MLIR_HOME=$HOME/llvm-install
export PATH=$MLIR_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MLIR_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MLIR_HOME/python_packages/mlir_core:$PYTHONPATH

#Verify it using this command
which mlir-opt

```

### Installing torch-mlir

```bash
# Using pip (recommended)
pip install torch-mlir

# Or from source
git clone https://github.com/llvm/torch-mlir
cd torch-mlir

git submodule update --init --recursive

# Start from a clean build dir
rm -rf build
mkdir build
cd build

# Configure: point to llvm-project/llvm, and attach torch-mlir as an external project
cmake -GNinja ../externals/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;stablehlo" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=../ \
  -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR=../externals/stablehlo \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python) \
  -DLLVM_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host

# Build torch-mlir Python modules
cmake --build . --target TorchMLIRPythonModules -j$(sysctl -n hw.logicalcpu)

```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anubhavkhajuria/phase_ordering.git
   cd phase_ordering
   ```

2. **Verify structure:**
   ```bash
   ls -la experiment1/
   # Should show: alexnet_linalg.mlir, Pipeline.sh, main.c, stb_image.h, etc.
   ```

3. **Prepare ImageNet class labels:**
   ```bash
   # Download or create imagenet_classes.txt
   # Format: one class name per line (1000 lines total)
   # Example:
   # tench
   # goldfish
   # great_white_shark
   # ...
   ```

4. **Make pipeline scripts executable:**
   ```bash
   chmod +x AlexNet_to_LLVM-IR/Pipeline.sh
   ```

## Usage

### Pipeline Execution

Each folder contains a complete pipeline. Here's how to run them:

#### AlexNet_to_LLVM - Baseline Pipeline

```bash
cd AlexNet_to_LLVM

# Run the MLIR optimization pipeline
./Pipeline.sh

# This will generate:
# - step1.mlir (after canonicalization + CSE)
# - step2.mlir (after bufferization)
# - step3.mlir (after loop conversion)
# - alexnet_llvm_dialect.mlir (LLVM dialect)
# - alexnet.ll (LLVM IR)
# - alexnet.o (object file)
```

#### Optimized_Pipeline_1 - Alternative Ordering

```bash
cd Optimized_Pipeline_1

# Run alternative pipeline
./Pipeline.sh

# Generates same output files but with different optimization ordering
```

#### Optimized_Pipeline_2 - Third Variant

```bash
cd Optimized_Pipeline_2

# Run third variant
./Pipeline.sh
```

### Compilation and Running

After running the pipeline, compile and execute the inference:

```bash
# Method 1: Using Clang (recommended for MLIR runtime)
clang -march=native main.c alexnet.o    -lmlir_c_runner_utils     -lmlir_runner_utils -no-pie     -lm     
-o alexnet_infer

# Method 2: Using GCC (simpler, if MLIR runtime is in system path)
gcc -march=native -O2 main.c alexnet.o \
    -lmlir_c_runner_utils \
    -lmlir_runner_utils \
    -lm \
    -o alexnet_infer

# Method 3: Using GCC with explicit library path
gcc -march=native -O2 main.c alexnet.o \
    -L/usr/local/lib \
    -L/path/to/llvm-project/build/lib \
    -lmlir_c_runner_utils \
    -lmlir_runner_utils \
    -lm \
    -Wl,-rpath,/path/to/llvm-project/build/lib \
    -o alexnet_infer

# Run inference on an image
./alexnet_infer path/to/your/image.jpg
```

### Example with Test Image

```bash
# Download a test image
wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg -O cat.jpg

# Run inference
./alexnet_infer cat.jpg
```

**Expected Output:**
```
I am AlexNet and i was highly influential that popularized the use of neural networks
Input: 224x224x3
Classes: 1000

Loaded 1000 ImageNet class labels

Loading image: cat.jpg
Original image: 481x640 pixels, 3 channels
Resizing to 224x224...
Take a breakk!!!! Have a KITKAT.... Normalization in progress......

 Funtime is over. Normalization completed :-) 

You can chill again.... Inferencing is progress!!!!!!
Time taken for inference: 0.234567 seconds

Break over. Run another test........Inference completed!!!!!

Top-10 Predictions:
 1. Class  281 (tabby_cat                    ): 0.4532 (45.32%)
 2. Class  282 (tiger_cat                    ): 0.2134 (21.34%)
 3. Class  285 (Egyptian_cat                 ): 0.1234 (12.34%)
 ...

Not so confidence Metrics:

  Top-1 confidence: 45.32%
  Top-5 confidence: 87.65%

  moderate confidence
                Everythin is over!!!!
```

## Understanding the Pipeline

### Stage 1: Canonicalization & Common Subexpression Elimination
```bash
mlir-opt alexnet_linalg.mlir \
  --canonicalize \
  --cse \
  -o step1.mlir
```
- **Canonicalize**: Simplifies operations to canonical forms
- **CSE**: Eliminates redundant computations
- **Output**: `step1.mlir` - optimized high-level representation

### Stage 2: Bufferization
```bash
mlir-opt step1.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o step2.mlir
```
- Converts tensor operations to buffer operations
- Allocates memory explicitly
- **Output**: `step2.mlir` - buffer-based representation

### Stage 3: Loop Conversion
```bash
mlir-opt step2.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  -o step3.mlir
```
- Converts high-level Linalg ops to explicit loops
- Converts structured control flow to control flow
- **Output**: `step3.mlir` - explicit loops

### Stage 4: Lowering to LLVM
```bash
mlir-opt step3.mlir \
  --lower-affine \
  --expand-strided-metadata \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o alexnet_llvm_dialect.mlir
```
- Lowers all MLIR dialects to LLVM dialect
- **Output**: `alexnet_llvm_dialect.mlir`

### Stage 5: LLVM IR Generation
```bash
mlir-translate --mlir-to-llvmir alexnet_llvm_dialect.mlir > alexnet.ll
```
- Converts MLIR LLVM dialect to standard LLVM IR
- **Output**: `alexnet.ll` - can be optimized with `opt`

### Stage 6: Object File Generation
```bash
llc -filetype=obj -relocation-model=pic alexnet.ll -o alexnet.o
```
- Compiles LLVM IR to machine code
- **Output**: `alexnet.o` - linkable object file

## Experiments

### Comparing Different Phase Orderings

Run all three experiments and compare:

```bash
# AlexNet_to_LLVM-IR
cd AlexNet_to_LLVM-IR
./Pipeline.sh
gcc -march=native -O2 main.c alexnet.o -lmlir_c_runner_utils -lmlir_runner_utils -lm -o exp1_infer
time ./exp1_infer ../test_images/dog.jpg > exp1_results.txt

# Optimized_Pipeline_1 
cd ../Optimized_Pipeline_1
./Pipeline.sh
gcc -march=native -O2 main.c alexnet.o -lmlir_c_runner_utils -lmlir_runner_utils -lm -o exp2_infer
time ./exp2_infer ../test_images/dog.jpg > exp2_results.txt

# Optimized_Pipeline_2
cd ../Optimized_Pipeline_1
./Pipeline.sh
gcc -march=native -O2 main.c alexnet.o -lmlir_c_runner_utils -lmlir_runner_utils -lm -o exp3_infer

# Compare results
cd ..
echo "=== Inference Time Comparison ==="
grep "Time taken" experiment*/exp*_results.txt
```

### Analyzing Generated Code

```bash
# Compare LLVM IR sizes
wc -l experiment1/alexnet.ll experiment2/alexnet.ll experiment3/alexnet.ll

# Compare object file sizes
ls -lh experiment1/alexnet.o experiment2/alexnet.o experiment3/alexnet.o

# View LLVM IR optimizations
llvm-dis alexnet.ll -o alexnet_readable.ll
less alexnet_readable.ll

# Disassemble object file
objdump -d alexnet.o | less
```

### Custom Pipeline Experiments

Modify `Pipeline.sh` to test different orderings:

```bash
#!/bin/bash

# Your custom ordering
mlir-opt alexnet_linalg.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --canonicalize \
  --cse \
  -o step1_custom.mlir

mlir-opt step1_custom.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  -o step2_custom.mlir

# Continue with lowering...
```

## Troubleshooting

### Common Issues

#### 1. MLIR Runtime Libraries Not Found
```bash
# Error: cannot find -lmlir_c_runner_utils
# Solution: Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/llvm-project/build/lib:$LD_LIBRARY_PATH

# Or use -Wl,-rpath during compilation
gcc main.c alexnet.o -Wl,-rpath,/path/to/llvm-project/build/lib -lmlir_c_runner_utils -lmlir_runner_utils -lm -o alexnet_infer
```

#### 2. Image Loading Fails
```bash
# Error: Failed to load image
# Check: Image format (should be JPG)
file your_image.jpg

# Check: Image path
ls -la your_image.jpg

# Check: stb_image.h is present
ls -la stb_image.h
```

#### 3. mlir-opt Not Found
```bash
# Add LLVM/MLIR to PATH
export PATH=/path/to/llvm-project/build/bin:$PATH

# Verify
which mlir-opt
mlir-opt --version
```

#### 4. Undefined Reference to `alexnet`
```bash
# Error: undefined reference to `alexnet'
# Solution: Ensure alexnet.o is generated and linked
ls -la alexnet.o
gcc main.c alexnet.o -lmlir_c_runner_utils -lmlir_runner_utils -lm -o alexnet_infer
```

#### 5. ImageNet Classes Not Loading
```bash
# Create or download imagenet_classes.txt
# Place in same directory or parent directory
cp imagenet_classes.txt experiment1/
# Or: cp imagenet_classes.txt experiment1/../
```

### Debugging Tips

```bash
# Check MLIR at each stage
mlir-opt --version
mlir-opt step1.mlir --verify-diagnostics

# Verify LLVM IR
llvm-as alexnet.ll -o /dev/null  # Check for syntax errors

# Check for missing symbols
nm alexnet.o | grep alexnet

# Verbose compilation
gcc -v main.c alexnet.o -lmlir_c_runner_utils -lmlir_runner_utils -lm -o alexnet_infer

# Run with debug output
./alexnet_infer test.jpg 2>&1 | tee debug.log
```

## Results and Analysis

### Metrics to Compare

1. **Inference Time**: Check `Time taken for inference` output
2. **Code Size**: Compare `alexnet.o` sizes
3. **LLVM IR Lines**: Count lines in `alexnet.ll`
4. **Memory Usage**: Use `valgrind` or `/usr/bin/time -v`



## Reference Resources

- [MLIR Documentation](https://mlir.llvm.org/)
- [torch-mlir Documentation](https://github.com/llvm/torch-mlir)
- [LLVM Optimization Guide](https://llvm.org/docs/Passes.html)
- [Phase Ordering Research Papers](https://scholar.google.com/scholar?q=compiler+phase+ordering)


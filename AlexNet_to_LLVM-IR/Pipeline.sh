#!/bin/bash


echo "Stage 1: Initial canonicalization..."
mlir-opt alexnet_linalg.mlir \
  --canonicalize \
  --cse \
  -o step1.mlir

mlir-opt step1.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o step2.mlir

mlir-opt step2.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  -o step3.mlir

mlir-opt step2.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  -o step3.mlir


mlir-opt step3.mlir \
 --lower-affine \
 --expand-strided-metadata \
 --finalize-memref-to-llvm \
 --convert-arith-to-llvm \
 --convert-func-to-llvm \
 --convert-cf-to-llvm \
 --reconcile-unrealized-casts \
 -o alexnet_llvm_dialect.mlir

 mlir-translate --mlir-to-llvmir alexnet_llvm_dialect.mlir > alexnet.ll

llc -filetype=obj -relocation-model=pic alexnet.ll -o alexnet.o

echo " use this command to run the model: clang  -march=native main.c alexnet.o   -L/data/anubhav/llvm-project/build/lib   -lmlir_c_runner_utils -lmlir_runner_utils -lm   -o alexnet_infer
"
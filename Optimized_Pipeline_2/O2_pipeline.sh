#!/bin/bash

# VECTORIZATION 

echo "=== AGGRESSIVE VECTORIZATION PIPELINE ==="

# Stage 1: Initial cleanup
echo "Stage 1: Initial canonicalization..."
mlir-opt alexnet_linalg.mlir \
  --canonicalize \
  --cse \
  -o vec_step1.mlir

# Stage 2: Prepare for vectorization - fuse operations
echo "Stage 2: Fuse elementwise operations..."
mlir-opt vec_step1.mlir \
  --linalg-fuse-elementwise-ops \
  --linalg-fold-unit-extent-dims \
  --canonicalize \
  --cse \
  -o vec_step2.mlir

# Stage 3: Generalize and prepare for tiling
echo "Stage 3: Generalize named ops..."
mlir-opt vec_step2.mlir \
  --linalg-generalize-named-ops \
  --canonicalize \
  -o vec_step3.mlir

# Stage 4: Bufferization (moved earlier, before vectorization)
echo "Stage 4: Bufferize..."
mlir-opt vec_step3.mlir \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --canonicalize \
  -o vec_step4_bufferized.mlir

# Stage 4b: Handle deallocations immediately
echo "Stage 4b: Lower deallocations..."
mlir-opt vec_step4_bufferized.mlir \
  --buffer-deallocation-pipeline \
  --canonicalize \
  -o vec_step4_dealloc.mlir

# Stage 5: Convert linalg to loops
echo "Stage 5: Lower linalg to loops..."
mlir-opt vec_step4_dealloc.mlir \
  --convert-linalg-to-loops \
  --canonicalize \
  --cse \
  -o vec_step5_loops.mlir

# Stage 6: Affine loop optimizations
echo "Stage 6: Affine optimizations..."
mlir-opt vec_step5_loops.mlir \
  --loop-invariant-code-motion \
  --affine-loop-fusion \
  --affine-loop-tile="tile-size=32" \
  --canonicalize \
  --cse \
  -o vec_step6_affine_opt.mlir

# Stage 7: SCF optimizations
echo "Stage 7: SCF optimizations..."
mlir-opt vec_step6_affine_opt.mlir \
  --scf-for-loop-peeling \
  --canonicalize \
  -o vec_step7_scf_opt.mlir

# Stage 8: Lower SCF to CF
echo "Stage 8: Lower SCF to CF..."
mlir-opt vec_step7_scf_opt.mlir \
  --convert-scf-to-cf \
  --canonicalize \
  -o vec_step8_cf.mlir

# Stage 9: Lower affine and normalize memrefs
echo "Stage 9: Lower affine..."
mlir-opt vec_step8_cf.mlir \
  --lower-affine \
  --normalize-memrefs \
  --memref-expand \
  --fold-memref-alias-ops \
  --canonicalize \
  -o vec_step9_lowered.mlir

# Stage 10: Expand strided metadata and lower affine again (for linearize_index)
echo "Stage 10: Expand metadata and lower affine..."
mlir-opt vec_step9_lowered.mlir \
  --expand-strided-metadata \
  --lower-affine \
  --canonicalize \
  -o vec_step10_expanded.mlir

# Stage 11: Final lowering to LLVM dialect
echo "Stage 11: Convert to LLVM dialect..."
mlir-opt vec_step10_expanded.mlir \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  --canonicalize \
  -o vec_step11_llvm.mlir

# Stage 12: Translate to LLVM IR
echo "Stage 12: Translate to LLVM IR..."
mlir-translate --mlir-to-llvmir vec_step11_llvm.mlir > alexnet_vectorized.ll

# Stage 13: LLVM optimizations with aggressive vectorization
echo "Stage 13: LLVM optimizations..."
opt --passes="default<O3>,loop-vectorize,slp-vectorizer,load-store-vectorizer" \
  alexnet_vectorized.ll -o alexnet_vectorized.bc

# Stage 14: Code generation
echo "Stage 14: Generate native code..."
llc -O3 \
  -march=x86-64 \
  -mcpu=native \
  -relocation-model=pic \
  -enable-unsafe-fp-math \
  -fp-contract=fast \
  -mattr=+avx2,+fma,+f16c \
  alexnet_vectorized.bc -o alexnet_vectorized.s

echo "=== AGGRESSIVE VECTORIZATION COMPLETE ==="
echo ""
echo "To compile and test:"
echo "  clang -march=native -fPIC -no-pie test.c alexnet_vectorized.s \\"
echo "    -L/data/anubhav/llvm-project/build/lib \\"
echo "    -lmlir_c_runner_utils -lmlir_runner_utils -lm \\"
echo "    -o alexnet_test"

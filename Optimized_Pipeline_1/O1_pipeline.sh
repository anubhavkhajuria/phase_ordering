#!/bin/bash

# Stage 1: Initial cleanup and canonicalization
echo "Stage 1: Canonicalization and CSE..."
mlir-opt alexnet_linalg.mlir \
  --canonicalize \
  --cse \
  -o step1_canon.mlir

# Stage 2: High-level Linalg optimizations
echo "Stage 2: Linalg optimizations..."
mlir-opt step1_canon.mlir \
  --linalg-fuse-elementwise-ops \
  --linalg-fold-unit-extent-dims \
  --canonicalize \
  --cse \
  -o step2_linalg_opt.mlir

# Stage 3: Vectorization preparation and tiling
echo "Stage 3: Tiling and vectorization prep..."
mlir-opt step2_linalg_opt.mlir \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --canonicalize \
  -o step3_generalized.mlir


# Stage 4: Bufferization
echo "Stage 4: Bufferization..."
mlir-opt step3_generalized.mlir \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --canonicalize \
  -o step4_bufferized.mlir

# Stage 4b: Lower deallocations 
echo "Stage 4b: Lower deallocations..."
mlir-opt step4_bufferized.mlir \
  --buffer-deallocation-pipeline \
  --canonicalize \
  -o step4_dealloc.mlir

# Stage 5: Convert linalg to loops with optimizations
echo "Stage 5: Convert linalg to loops..."
mlir-opt step4_dealloc.mlir \
  --convert-linalg-to-loops \
  --canonicalize \
  --cse \
  -o step5_loops.mlir

# Stage 6: Loop optimizations 
echo "Stage 6: Loop optimizations..."
mlir-opt step5_loops.mlir \
  --loop-invariant-code-motion \
  --affine-loop-fusion \
  --affine-loop-tile="tile-sizes=32 tile-sizes=32" \
  --canonicalize \
  -o step6_loop_opt.mlir

# Stage 7: SCF optimizations
echo "Stage 7: SCF optimizations..."
mlir-opt step6_loop_opt.mlir \
  --scf-for-loop-peeling \
  --scf-for-loop-canonicalization \
  --canonicalize \
  -o step7_scf_opt.mlir

# Stage 8: Convert SCF to CF
echo "Stage 8: Convert SCF to CF..."
mlir-opt step7_scf_opt.mlir \
  --convert-scf-to-cf \
  --canonicalize \
  -o step8_cf.mlir

# Stage 9: Affine and memref optimizations
echo "Stage 9: Affine and memref optimizations..."
mlir-opt step8_cf.mlir \
  --lower-affine \
  --normalize-memrefs \
  --memref-expand \
  --fold-memref-alias-ops \
  --canonicalize \
  -o step9_affine_lowered.mlir

# Stage 10: Arithmetic optimizations
echo "Stage 10: Arithmetic optimizations..."
mlir-opt step9_affine_lowered.mlir \
  --arith-expand \
  --canonicalize \
  --cse \
  -o step10_arith_opt.mlir

# Stage 11: Final lowering to LLVM dialect
echo "Stage 11: Lower to LLVM dialect..."
mlir-opt step10_arith_opt.mlir \
  --lower-affine \
  --expand-strided-metadata \
  --finalize-memref-to-llvm \
  --lower-affine \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --reconcile-unrealized-casts \
  --canonicalize \
  -o step11_llvm_dialect.mlir


# Stage 12: Translate to LLVM IR
echo "Stage 12: Translate to LLVM IR..."
mlir-translate --mlir-to-llvmir step11_llvm_dialect.mlir > alexnet.ll

# Stage 13: LLVM optimizations
echo "Stage 13: LLVM optimization passes..."
opt 
  -passes="loop-vectorize,slp-vectorizer,load-store-vectorizer" \
  alexnet.ll -o alexnet_opt.bc

# Stage 14: Convert to assembly or object file
echo "Stage 14: Generate native code..."
llc -O3 \
  -march=x86-64 \
  -mcpu=native \
  -enable-unsafe-fp-math \
  -mattr=+avx2,+fma \
  alexnet_opt.bc -o alexnet.s

# Optional: Create object file
# llc -O3 -march=x86-64 -mcpu=native -filetype=obj alexnet_opt.bc -o alexnet.o

echo "Compilation complete!"

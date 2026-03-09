# Lab 2: Basic CUDA Programming

**Accelerator-Centric Computing Ecosystems (XM_0171)**  
**Target hardware:** DAS-5 `gpunode,TitanX` (Maxwell, sm_52)

## Task progression

1. **Task 1 — Memory management:** Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`.

2. **Task 2 — Kernel implementation:** Write the matmul dot product.
   With N=16 and a single block, `Test PASSED` should appear.

3. **Task 3 — Multi-block launch:** Compute `gridDim` with ceiling division to cover
   the full 512×512 matrix. Includes CUDA event timing and speedup reporting.

4. **Task 4 — Grid-stride loop:** With a fixed small grid (4×4 blocks), each thread
   must process multiple elements via a stride loop.

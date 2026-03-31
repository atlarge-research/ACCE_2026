/**
 * LAB 2 - TASK 2: Implement the Matmul Kernel
 * =============================================
 * Accelerator-Centric Computing Ecosystems (XM_0171)
 *
 * Memory management is now complete. The kernel function matmul_kernel
 * exists but its body is empty.
 *
 * Your job: implement the kernel body (TODO 2a through 2c).
 *
 * For a small matrix (16x16), the result should be correct ("Test PASSED").
 * For a larger matrix (e.g. 512x512), only the top-left 16x16 block will
 * be correct because we only launch a single block. Task 3 fixes this.
 *
 * Compile:  nvcc task2_matmul.cu -o task2_matmul
 * Run:      sbatch run_task2.sh
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// GPU kernel -- YOU IMPLEMENT THIS
// ---------------------------------------------------------------------------
__global__ void matmul_kernel(float *A, float *B, float *C, int N, int K,
                              int M) {
  // TODO 2a: Compute the global row and col index for this thread.
  //          row = blockIdx.y * blockDim.y + threadIdx.y
  //          col = blockIdx.x * blockDim.x + threadIdx.x

  // TODO 2b: Add a bounds check -- only compute if row < N and col < M

  // TODO 2c: Compute the dot product for C[row][col].
  //          Loop over the K dimension, accumulating:
  //              sum += A[row * K + i] * B[i * M + col]
  //          Then write: C[row * M + col] = sum
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------
void matmul_cpu(float *A, float *B, float *C, int N, int K, int M) {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < M; col++) {
      float sum = 0.0f;
      for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * M + col];
      }
      C[row * M + col] = sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Verification helper
// ---------------------------------------------------------------------------
int verify(float *gpu_C, float *cpu_C, int N, int M) {
  int errors = 0;
  for (int i = 0; i < N * M; i++) {
    if (fabsf(gpu_C[i] - cpu_C[i]) > 1e-2f) {
      if (errors < 5) {
        printf("  Mismatch at [%d,%d]: GPU = %f, CPU = %f\n", i / M, i % M,
               gpu_C[i], cpu_C[i]);
      }
      errors++;
    }
  }
  if (errors > 5)
    printf("  ... and %d more mismatches\n", errors - 5);
  return errors == 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  // Try with 16x16 first (single block suffices).
  // Once your kernel works, change to 512 and observe partial results.
  int N = 16, K = 16, M = 16;

  size_t size_A = N * K * sizeof(float);
  size_t size_B = K * M * sizeof(float);
  size_t size_C = N * M * sizeof(float);

  // Host allocation
  float *h_A = (float *)malloc(size_A);
  float *h_B = (float *)malloc(size_B);
  float *h_C = (float *)malloc(size_C);
  float *h_C_ref = (float *)malloc(size_C);

  // Initialise
  for (int i = 0; i < N * K; i++)
    h_A[i] = (float)(i % 7);
  for (int i = 0; i < K * M; i++)
    h_B[i] = (float)(i % 5);
  memset(h_C, 0, size_C);

  // CPU reference
  matmul_cpu(h_A, h_B, h_C_ref, N, K, M);

  // Device allocation
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // Copy inputs to device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // Launch kernel -- single block of 16x16 threads
  dim3 blockDim(16, 16);
  dim3 gridDim(1, 1);
  printf("Launch config: grid(%d,%d), block(%d,%d)\n", gridDim.x, gridDim.y,
         blockDim.x, blockDim.y);
  printf("Total threads: %d\n",
         gridDim.x * gridDim.y * blockDim.x * blockDim.y);

  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaDeviceSynchronize();

  // Copy result back
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // Verify
  printf("Matrix size: %d x %d x %d\n", N, K, M);
  if (verify(h_C, h_C_ref, N, M)) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}

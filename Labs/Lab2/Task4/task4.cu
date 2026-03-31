/**
 * LAB 2 - TASK 4: Grid-Stride Loop
 * ==================================
 * Accelerator-Centric Computing Ecosystems (XM_0171)
 *
 * The launch configuration is now FIXED to a small grid (4x4 blocks of
 * 16x16 threads = 4096 threads total). The matrix is 512x512 = 262,144
 * elements; far more than the number of threads.
 *
 * Your job:
 *   TODO 4a: Convert the matmul kernel to use a 2D grid-stride loop so
 *            that each thread processes multiple output elements.
 *
 * Do NOT change the launch configuration.
 *
 * Compile:  nvcc task4_matmul.cu -o task4_matmul
 * Run:      sbatch run_task4.sh
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---------------------------------------------------------------------------
// GPU kernel with grid-stride loop -- YOU IMPLEMENT THIS
// ---------------------------------------------------------------------------
__global__ void matmul_kernel_stride(float *A, float *B, float *C, int N, int K,
                                     int M) {
  // TODO 4a: Implement a 2D grid-stride loop.
  //
  // The idea:
  //   - Compute the starting row and col from blockIdx/threadIdx as before.
  //   - Compute the stride in each dimension:
  //       stride_row = gridDim.y * blockDim.y   (total threads in y)
  //       stride_col = gridDim.x * blockDim.x   (total threads in x)
  //   - Use nested for-loops that advance by the stride:
  //
  //       for (int row = <start_row>; row < N; row += stride_row) {
  //           for (int col = <start_col>; col < M; col += stride_col) {
  //               // dot product for C[row][col]
  //           }
  //       }
  //
  // Hint: the inner dot-product loop over K is identical to Task 2.
  //       The only change is wrapping it in the stride loops instead of
  //       using a simple if-guard.
}

// ---------------------------------------------------------------------------
// Non-strided kernel (from Task 3, for comparison)
// ---------------------------------------------------------------------------
__global__ void matmul_kernel(float *A, float *B, float *C, int N, int K,
                              int M) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * M + col];
    }
    C[row * M + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// CPU reference
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
// Verification
// ---------------------------------------------------------------------------
int verify(float *gpu_C, float *cpu_C, int total) {
  int errors = 0;
  for (int i = 0; i < total; i++) {
    if (fabsf(gpu_C[i] - cpu_C[i]) > 1e-2f) {
      if (errors < 5) {
        printf("  Mismatch at index %d: GPU = %f, CPU = %f\n", i, gpu_C[i],
               cpu_C[i]);
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
  int N = 512, K = 512, M = 512;

  size_t size_A = N * K * sizeof(float);
  size_t size_B = K * M * sizeof(float);
  size_t size_C = N * M * sizeof(float);

  // Host allocation
  float *h_A = (float *)malloc(size_A);
  float *h_B = (float *)malloc(size_B);
  float *h_C = (float *)malloc(size_C);
  float *h_C_ref = (float *)malloc(size_C);

  // Initialise
  srand(42);
  for (int i = 0; i < N * K; i++)
    h_A[i] = (float)(rand() % 10);
  for (int i = 0; i < K * M; i++)
    h_B[i] = (float)(rand() % 10);

  // CPU reference
  struct timespec cpu_start, cpu_end;
  clock_gettime(CLOCK_MONOTONIC, &cpu_start);
  matmul_cpu(h_A, h_B, h_C_ref, N, K, M);
  clock_gettime(CLOCK_MONOTONIC, &cpu_end);
  double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                  (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf("CPU time:          %.3f ms\n\n", cpu_ms);

  // Device allocation
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // =====================================================================
  // Fixed launch config -- DO NOT CHANGE
  // 4x4 blocks of 16x16 threads = 4096 threads total
  // =====================================================================
  dim3 blockDim(16, 16);
  dim3 gridDim(4, 4);

  printf("Fixed launch config: grid(%d,%d), block(%d,%d)\n", gridDim.x,
         gridDim.y, blockDim.x, blockDim.y);
  printf("Total threads:       %d  (matrix elements: %d)\n\n",
         gridDim.x * gridDim.y * blockDim.x * blockDim.y, N * M);

  // =====================================================================
  // Run 1: Non-strided kernel (from Task 3) -- expect partial results
  // =====================================================================
  cudaMemset(d_C, 0, size_C);

  cudaEvent_t k0, k1;
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);

  cudaEventRecord(k0);
  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  float non_stride_ms = 0;
  cudaEventElapsedTime(&non_stride_ms, k0, k1);

  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  printf("--- Non-strided kernel (only covers top-left 64x64) ---\n");
  printf("Kernel time:       %.3f ms\n", non_stride_ms);
  if (verify(h_C, h_C_ref, N * M)) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED (expected -- not enough threads)\n");
  }
  printf("\n");

  // =====================================================================
  // Run 2: Grid-stride kernel -- should cover entire matrix
  // =====================================================================
  cudaMemset(d_C, 0, size_C);

  cudaEvent_t k2, k3;
  cudaEventCreate(&k2);
  cudaEventCreate(&k3);

  cudaEventRecord(k2);
  matmul_kernel_stride<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaEventRecord(k3);
  cudaEventSynchronize(k3);

  float stride_ms = 0;
  cudaEventElapsedTime(&stride_ms, k2, k3);

  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  printf("--- Grid-stride kernel ---\n");
  printf("Kernel time:       %.3f ms\n", stride_ms);
  if (verify(h_C, h_C_ref, N * M)) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  // =====================================================================
  // Summary
  // =====================================================================
  printf("\n--- Comparison ---\n");
  printf("CPU:               %.3f ms\n", cpu_ms);
  printf("Grid-stride GPU:   %.3f ms\n", stride_ms);
  printf("Speedup (stride):  %.2fx over CPU\n", cpu_ms / stride_ms);

  // Cleanup
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(k2);
  cudaEventDestroy(k3);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}

/**
 * LAB 2 - TASK 3: Multi-Block Launch and Timing
 * ===============================================
 * Accelerator-Centric Computing Ecosystems (XM_0171)
 *
 * The kernel works for a 16x16 region. Time to make it work for full-size
 * matrices and measure performance.
 *
 * Your job:
 *   TODO 3a: Fix the grid dimensions using ceiling division so that
 *            enough blocks are launched to cover the entire matrix.
 *   TODO 3b: Observe the timing output and speedup.
 *
 * (At home) Experiment with different blockDim values: 8x8, 16x16, 32x32.
 *
 * Compile:  nvcc task3_matmul.cu -o task3_matmul
 * Run:      sbatch run_task3.sh
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---------------------------------------------------------------------------
// GPU kernel (complete from Task 2)
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
// Verification (checks first num_check elements)
// ---------------------------------------------------------------------------
int verify(float *gpu_C, float *cpu_C, int total, int num_check) {
  int errors = 0;
  for (int i = 0; i < num_check && i < total; i++) {
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
  memset(h_C, 0, size_C);

  // =====================================================================
  // CPU timing
  // =====================================================================
  struct timespec cpu_start, cpu_end;
  clock_gettime(CLOCK_MONOTONIC, &cpu_start);
  matmul_cpu(h_A, h_B, h_C_ref, N, K, M);
  clock_gettime(CLOCK_MONOTONIC, &cpu_end);
  double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                  (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf("CPU time:          %.3f ms\n", cpu_ms);

  // =====================================================================
  // GPU setup
  // =====================================================================
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // ----- Transfer timing (host -> device) -----
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  cudaEventRecord(t0);
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float h2d_ms = 0;
  cudaEventElapsedTime(&h2d_ms, t0, t1);
  printf("H->D transfer:     %.3f ms\n", h2d_ms);

  // =====================================================================
  // Kernel launch configuration
  // =====================================================================
  dim3 blockDim(16, 16);

  // TODO 3a: Replace the grid dimensions below.
  //          You need enough blocks to cover the full N x M output matrix.
  //          Use ceiling division:
  //              gridDim.x = (M + blockDim.x - 1) / blockDim.x   (columns)
  //              gridDim.y = (N + blockDim.y - 1) / blockDim.y   (rows)
  //
  //          Currently this launches only 1 block -- fix it!
  dim3 gridDim(1, 1);

  printf("Launch config:     grid(%d,%d), block(%d,%d)\n", gridDim.x, gridDim.y,
         blockDim.x, blockDim.y);
  printf("Total threads:     %d\n",
         gridDim.x * gridDim.y * blockDim.x * blockDim.y);

  // ----- Kernel timing -----
  cudaEvent_t k0, k1;
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);

  cudaEventRecord(k0);
  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);
  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, k0, k1);
  printf("Kernel time:       %.3f ms\n", kernel_ms);

  // ----- Transfer timing (device -> host) -----
  cudaEvent_t t2, t3;
  cudaEventCreate(&t2);
  cudaEventCreate(&t3);

  cudaEventRecord(t2);
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
  cudaEventRecord(t3);
  cudaEventSynchronize(t3);
  float d2h_ms = 0;
  cudaEventElapsedTime(&d2h_ms, t2, t3);
  printf("D->H transfer:     %.3f ms\n", d2h_ms);

  // =====================================================================
  // Performance summary
  // =====================================================================
  float total_gpu_ms = h2d_ms + kernel_ms + d2h_ms;
  printf("\n--- Performance Summary ---\n");
  printf("Matrix size:       %d x %d x %d\n", N, K, M);
  printf("CPU time:          %.3f ms\n", cpu_ms);
  printf("GPU kernel only:   %.3f ms\n", kernel_ms);
  printf("GPU end-to-end:    %.3f ms (transfers + kernel)\n", total_gpu_ms);
  printf("Speedup (kernel):  %.2fx\n", cpu_ms / kernel_ms);
  printf("Speedup (e2e):     %.2fx\n", cpu_ms / total_gpu_ms);

  // Verify
  if (verify(h_C, h_C_ref, N * M, N * M)) {
    printf("\nTest PASSED\n");
  } else {
    printf("\nTest FAILED\n");
  }

  // Cleanup
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}

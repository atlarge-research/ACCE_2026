#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---------------------------------------------------------------------------
// GPU kernel with grid-stride loop -- YOU IMPLEMENT THIS
// ---------------------------------------------------------------------------
__global__ void matmul_kernel_stride(float *A, float *B, float *C, int N, int K,
                                     int M) {
  int start_row = blockIdx.y * blockDim.y + threadIdx.y;
  int start_col = blockIdx.x * blockDim.x + threadIdx.x;

  int stride_row = gridDim.y * blockDim.y;
  int stride_col = gridDim.x * blockDim.x;

  for (int row = start_row; row < N; row += stride_row) {
    for (int col = start_col; col < M; col += stride_col) {
      float sum = 0.0f;
      for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * M + col];
      }
      C[row * M + col] = sum;
    }
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
int main(int argc, char **argv) {
  // Check command-line arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s blk_dim\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Parse grid and block dimensions from command line
  int blk_dim = atoi(argv[1]);

  if (blk_dim <= 0) {
    fprintf(stderr, "blk_dim must be positive integers.\n");
    exit(EXIT_FAILURE);
  }

  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  if (blk_dim * blk_dim > maxThreadsPerBlock) {
    fprintf(stderr,
            "Error: block size %d×%d = %d exceeds max 1024 threads per block\n",
            blk_dim, blk_dim, blk_dim * blk_dim);
    exit(EXIT_FAILURE);
  }

  int N = 1024, K = 1024, M = 1024;

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

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  // 2D block: blk_dim x blk_dim threads
  dim3 blockDim(blk_dim, blk_dim);
  int threads_per_block = blk_dim * blk_dim;

  int maxThreadsPerSM;
  cudaDeviceGetAttribute(&maxThreadsPerSM,
                         cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  int total_blocks = numSMs * maxThreadsPerSM / threads_per_block;

  int grid_side = (int)ceilf(sqrtf((float)total_blocks));
  dim3 gridDim(grid_side, grid_side);

  printf("Fixed launch config: grid(%d,%d), block(%d,%d)\n", gridDim.x,
         gridDim.y, blockDim.x, blockDim.y);
  printf("Total threads:       %d  (matrix elements: %d)\n\n",
         gridDim.x * gridDim.y * blockDim.x * blockDim.y, N * M);

  cudaMemset(d_C, 0, size_C);

  cudaEvent_t k0, k1;
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);

  // Warm up run
  matmul_kernel_stride<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaDeviceSynchronize();
  cudaMemset(d_C, 0, size_C);

  // Timed run
  cudaEventRecord(k0);
  matmul_kernel_stride<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  float stride_ms = 0;
  cudaEventElapsedTime(&stride_ms, k0, k1);

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
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}
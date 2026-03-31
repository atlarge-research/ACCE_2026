/**
 * LAB 2 - TASK 1: Memory Management
 * ==================================
 * Accelerator-Centric Computing Ecosystems (XM_0171)
 *
 * In this task you will complete the GPU memory management for a matrix
 * multiplication program. The host-side allocation, data initialisation,
 * and a test kernel are already provided.
 *
 * Your job: complete TODO 1a through 1f in main().
 *
 * The test kernel copies input matrix A into output matrix C on the GPU.
 * If your memory management is correct, h_C will match h_A after the
 * round-trip: host -> device -> kernel -> device -> host.
 *
 * Compile:  nvcc task1_matmul.cu -o task1_matmul
 * Run:      sbatch run_task1.sh
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Test kernel: copies A into C on the GPU.
// (We will write a real matmul kernel in Task 2.)
// ---------------------------------------------------------------------------
__global__ void copy_kernel(float *A, float *C, int num_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_elements) {
    C[i] = A[i];
  }
}

// ---------------------------------------------------------------------------
// Validation helpers (you can ignore these -- just read main)
// ---------------------------------------------------------------------------

// Returns 1 if both pointers are non-NULL, 0 otherwise. Prints status.
int check_alloc(float *d_A, float *d_C) {
  int ok = 1;
  printf("Step 1 - cudaMalloc:\n");
  if (d_A == NULL) {
    printf("  [INCOMPLETE] d_A is still NULL\n");
    ok = 0;
  } else {
    printf("  [OK] d_A allocated\n");
  }
  if (d_C == NULL) {
    printf("  [INCOMPLETE] d_C is still NULL\n");
    ok = 0;
  } else {
    printf("  [OK] d_C allocated\n");
  }
  printf("\n");
  if (!ok)
    printf("Cannot continue. Complete TODO 1a/1b first.\n");
  return ok;
}

// Reads back d_A and checks it matches h_A. Returns 1 on success.
int check_h2d(float *d_A, float *h_A, size_t size, int n) {
  float *tmp = (float *)malloc(size);
  memset(tmp, 0, size);
  cudaMemcpy(tmp, d_A, size, cudaMemcpyDeviceToHost);

  int ok = 1;
  for (int i = 0; i < n; i++) {
    if (fabsf(tmp[i] - h_A[i]) > 1e-5f) {
      ok = 0;
      break;
    }
  }
  free(tmp);

  printf("Step 2 - cudaMemcpy (Host -> Device):\n");
  if (ok)
    printf("  [OK] h_A successfully copied to d_A\n");
  else
    printf("  [INCOMPLETE] d_A does not contain h_A's data\n");
  printf("\n");
  return ok;
}

// Checks kernel launch for errors. Returns 1 on success.
int check_kernel(void) {
  cudaError_t err = cudaGetLastError();
  printf("Step 3 - Kernel launch:\n");
  if (err != cudaSuccess) {
    printf("  [ERROR] %s\n", cudaGetErrorString(err));
    printf("\n");
    return 0;
  }
  printf("  [OK] copy_kernel launched and completed\n\n");
  return 1;
}

// Checks h_C against h_A. h_C was initialised to -1, so sentinel
// detection tells us whether the D->H copy happened at all. Returns 1 on match.
int check_d2h(float *h_C, float *h_A, int n) {
  int match = 1, still_sentinel = 1;
  for (int i = 0; i < n; i++) {
    if (h_C[i] != -1.0f)
      still_sentinel = 0;
    if (fabsf(h_C[i] - h_A[i]) > 1e-5f)
      match = 0;
  }

  printf("Step 4 - cudaMemcpy (Device -> Host):\n");
  if (still_sentinel) {
    printf("  [INCOMPLETE] h_C still contains sentinel values (-1)\n");
    printf("\n");
    return 0;
  }
  if (match)
    printf("  [OK] d_C successfully copied back to h_C\n");
  else
    printf("  [PARTIAL] h_C modified but does not match expected values\n");
  printf("\n");
  return match;
}

// Prints the final verdict.
void print_result(int h2d_ok, int d2h_ok) {
  printf("Step 5 - cudaFree:\n");
  printf("  (Make sure you added cudaFree for both d_A and d_C.)\n\n");

  printf("==========================================\n");
  if (h2d_ok && d2h_ok) {
    printf("  ALL STEPS OK -- Memory round-trip PASSED!\n");
    printf("  h_A -> d_A -> kernel -> d_C -> h_C : verified.\n");
    printf("  You are ready for Task 2.\n");
  } else {
    printf("  INCOMPLETE -- review the [INCOMPLETE] steps above.\n");
  }
  printf("==========================================\n");
}

// ---------------------------------------------------------------------------
// Main -- this is where you work
// ---------------------------------------------------------------------------
int main() {
  int N = 16, M = 16;
  int num_elements = N * M;
  size_t size_A = num_elements * sizeof(float);
  size_t size_C = num_elements * sizeof(float);

  printf("=== TASK 1: Memory Management ===\n");
  printf("Matrix: %d x %d (%zu bytes)\n\n", N, M, size_A);

  // ----- Host allocation (done for you) -----
  float *h_A = (float *)malloc(size_A);
  float *h_C = (float *)malloc(size_C);
  for (int i = 0; i < num_elements; i++)
    h_A[i] = (float)(i + 1);
  for (int i = 0; i < num_elements; i++)
    h_C[i] = -1.0f; // sentinel

  // =====================================================================
  // STEP 1: Allocate GPU memory
  // =====================================================================
  float *d_A = NULL;
  float *d_C = NULL;

  // TODO 1a: Allocate GPU memory for d_A (size_A bytes)
  //          cudaMalloc();

  // TODO 1b: Allocate GPU memory for d_C (size_C bytes)
  //          cudaMalloc();

  if (!check_alloc(d_A, d_C)) {
    free(h_A);
    free(h_C);
    return 1;
  }

  // =====================================================================
  // STEP 2: Copy input data from host to device
  // =====================================================================

  // TODO 1c: Copy h_A to d_A
  //          cudaMemcpy();

  int h2d_ok = check_h2d(d_A, h_A, size_A, num_elements);

  // =====================================================================
  // STEP 3: Kernel launch (done for you, do not modify)
  // =====================================================================
  int tpb = 256;
  int nblocks = (num_elements + tpb - 1) / tpb;
  copy_kernel<<<nblocks, tpb>>>(d_A, d_C, num_elements);
  cudaDeviceSynchronize();
  check_kernel();

  // =====================================================================
  // STEP 4: Copy result from device to host
  // =====================================================================

  // TODO 1d: Copy d_C back to h_C
  //          cudaMemcpy();

  int d2h_ok = check_d2h(h_C, h_A, num_elements);

  // =====================================================================
  // STEP 5: Free device memory
  // =====================================================================

  // TODO 1e: Free d_A
  //          cudaFree();

  // TODO 1f: Free d_C
  //          cudaFree();

  // =====================================================================
  print_result(h2d_ok, d2h_ok);

  free(h_A);
  free(h_C);
  return 0;
}

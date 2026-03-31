#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#define TILE_DIM 16  // You can adjust the block size
#define CHECK_CUDA_ERR(call)                                           \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// ---------------------------------------------------------
// CUDA Kernel: Tiled Matrix-Matrix Multiplication
// C = A * B
// ---------------------------------------------------------
__global__ void matMulTiledKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) 
{
    // Declare shared memory
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    // Calculate the global coordinates corresponding to this thread
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Process all tiles one by one
    // Assume N is divisible by TILE_DIM
    for (int tile = 0; tile < (N / TILE_DIM); ++tile) {
        // Load from global memory to shared memory
        A_s[threadIdx.y][threadIdx.x] = A[row * N + (tile * TILE_DIM + threadIdx.x)];
        B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];

        // Wait for all threads to finish loading
        __syncthreads();

        // Perform computation in shared memory
        for (int i = 0; i < TILE_DIM; i++) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }

        // Wait for all threads to finish computing this tile
        __syncthreads();
    }

    // Write the result back to global memory
    C[row * N + col] = sum;
}

// ---------------------------------------------------------
// Perform matrix multiplication on the CPU for verification
// and compare with the GPU result
// ---------------------------------------------------------
void verifyResult(const float* A, const float* B, const float* C, int N)
{
    const double epsilon = 1.0e-3;  // Tolerance for error

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double cpuSum = 0.0;
            for (int k = 0; k < N; k++) {
                cpuSum += static_cast<double>(A[i * N + k]) *
                          static_cast<double>(B[k * N + j]);
            }
            double diff = std::fabs(cpuSum - C[i * N + j]);
            if (diff > epsilon) {
                std::cout << "Verification failed at (" << i << ", " << j << ")\n"
                          << "CPU = " << cpuSum << ", GPU = " << C[i * N + j]
                          << ", Error = " << diff << std::endl;
                return;
            }
        }
    }
    std::cout << "Verification PASSED!\n";
}

// ---------------------------------------------------------
// Main program: allocate memory, initialize data, call kernel,
// and verify results
// ---------------------------------------------------------
int main(int argc, char* argv[])
{
    // You can adjust the matrix size
    // It is recommended to make N divisible by blockDim (e.g., 16) for easier comparison
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return EXIT_FAILURE;
    }
    const int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return EXIT_FAILURE;
    }
    if (N % 16 != 0) {
        std::cerr << "Matrix size must be a multiple of 16." << std::endl;
        return EXIT_FAILURE;
    }

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize A, B
    srand(123);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_C, size));

    // Copy A, B to device
    CHECK_CUDA_ERR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Set block and grid dimensions
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N / block.x, N / block.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA_ERR(cudaEventRecord(start, 0));

    // Call kernel
    matMulTiledKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    // Record the stop event
    CHECK_CUDA_ERR(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    float elapsedTime;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&elapsedTime, start, stop));

    // Destroy CUDA events
    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

    // Wait for GPU to finish and check for errors
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Copy result C from device to host
    CHECK_CUDA_ERR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results on CPU
    verifyResult(h_A, h_B, h_C, N);
    std::cout << "[Tiled (shared memory) Version] Kernel Execution Time = " << elapsedTime << " ms\n";

    // Print the GOPs/s
    double gops = (2.0 * N * N * N) / (elapsedTime * 1e6);  // Convert to Giga Operations per second
    std::cout << "[Tiled (shared memory) Version] Performance = " << gops << " GOPs/s\n";

    // Free device memory
    CHECK_CUDA_ERR(cudaFree(d_A));
    CHECK_CUDA_ERR(cudaFree(d_B));
    CHECK_CUDA_ERR(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

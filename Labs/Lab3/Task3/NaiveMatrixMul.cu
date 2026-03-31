#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

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
// Naive Matrix-Matrix Multiplication Kernel
// C = A * B
// ---------------------------------------------------------
__global__ void matMulNaiveKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N)
{
    // Calculate the global coordinates corresponding to this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------
// Perform matrix multiplication on the CPU for verification,
// and compare with the GPU computation result
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
                std::cout << "Result verification failed at (" << i << ", " << j << ")\n"
                          << "CPU = " << cpuSum << ", GPU = " << C[i * N + j]
                          << ", Error = " << diff << std::endl;
                return;
            }
        }
    }
    std::cout << "Result verification passed!\n";
}

// ---------------------------------------------------------
// Main program: allocate memory, initialize data, call kernel, verify result
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
    float* h_C = (float*)malloc(size);  // To store the result of the naive version

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
    // The naive version can use the same block size for comparison
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // Set CUDA event to measure kernel execution time
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA_ERR(cudaEventRecord(start));

    // Call the naive version kernel
    matMulNaiveKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    // Stop timing
    CHECK_CUDA_ERR(cudaEventRecord(stop));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    // Calculate execution time (milliseconds)
    float milliseconds = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy back the result
    CHECK_CUDA_ERR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify the result on the CPU
    verifyResult(h_A, h_B, h_C, N);

    std::cout << "[Naive Version] Kernel Execution Time = "
              << milliseconds << " ms\n";

    // Print the GOPs/s
    double gops = (2.0 * N * N * N) / (milliseconds * 1e6);
    std::cout << "[Naive Version] Performance = "
              << gops << " GOPs/s\n";

    // Destroy CUDA event
    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

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

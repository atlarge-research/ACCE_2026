#include "block_mm_no_dataflow.h"

void load(DTYPE *in, DTYPE Block[BLOCK_SIZE][BLOCK_SIZE], int ib, int jb, int row_size) {
	for(int i = 0; i < BLOCK_SIZE; i++)
		for(int j = 0; j < BLOCK_SIZE; j++)
			Block[i][j] = in[(ib * BLOCK_SIZE + i) * row_size + jb * BLOCK_SIZE + j];  
}

void store(DTYPE *out, DTYPE Block[BLOCK_SIZE][BLOCK_SIZE], int ib, int jb, int row_size) {
    // store and add the partial sum back to the DRAM
	for(int i= 0; i < BLOCK_SIZE; i++)
		for(int j = 0; j < BLOCK_SIZE; j++)
			out[(ib * BLOCK_SIZE + i) * row_size + jb * BLOCK_SIZE + j] += Block[i][j];
}

void compute(DTYPE Block_A[BLOCK_SIZE][BLOCK_SIZE], DTYPE Block_B[BLOCK_SIZE][BLOCK_SIZE], DTYPE Block_C[BLOCK_SIZE][BLOCK_SIZE]) {
	
	for(int i = 0; i < BLOCK_SIZE; i++) {
		for(int j = 0; j < BLOCK_SIZE; j++) {
			// clear output mat
			Block_C[i][j] = 0;
			for(int k = 0; k < BLOCK_SIZE; k++)
				Block_C[i][j] += Block_A[i][k] * Block_B[k][j];
		}
	}
}

void blockmatmul(DTYPE *A, DTYPE *B, DTYPE *C) {
	#pragma HLS INTERFACE m_axi port=A bundle=gemm0 depth=N*M
	#pragma HLS INTERFACE m_axi port=B bundle=gemm1 depth=M*P
	#pragma HLS INTERFACE m_axi port=C bundle=gemm2 depth=N*P

	DTYPE Block_A[BLOCK_SIZE][BLOCK_SIZE];
	DTYPE Block_B[BLOCK_SIZE][BLOCK_SIZE];
	DTYPE Block_C[BLOCK_SIZE][BLOCK_SIZE];
	
	Block_i: for(int i = 0; i < N / BLOCK_SIZE; i++) 
		Block_j: for(int j = 0; j < P / BLOCK_SIZE; j++) 
			Block_k: for(int k = 0; k < M / BLOCK_SIZE; k++) {
				load(A, Block_A, i, k, M);
				load(B, Block_B, k, j, P);
				compute(Block_A, Block_B, Block_C);
				store(C, Block_C, i, j, P);
			}
			
}

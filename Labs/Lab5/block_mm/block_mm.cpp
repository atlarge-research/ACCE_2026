#include "block_mm.h"

void load(DTYPE *in, hls::stream<DTYPE> &stream, int ib, int jb, int row_size) {
	for(int i = 0; i < BLOCK_SIZE; i++)
		for(int j = 0; j < BLOCK_SIZE; j++)
			stream.write(in[(ib * BLOCK_SIZE + i) * row_size + jb * BLOCK_SIZE + j]);  
}

void store(DTYPE *out, hls::stream<DTYPE> &stream, int ib, int jb, int row_size) {
	// store and add the partial sum back to the DRAM
	for(int i= 0; i < BLOCK_SIZE; i++)
		for(int j = 0; j < BLOCK_SIZE; j++)
			out[(ib * BLOCK_SIZE + i) * row_size + jb * BLOCK_SIZE + j] += stream.read();
}

void compute(hls::stream<DTYPE> &streamA, hls::stream<DTYPE> &streamB, hls::stream<DTYPE> &streamC) {

	DTYPE Block_A[BLOCK_SIZE][BLOCK_SIZE];
	DTYPE Block_B[BLOCK_SIZE][BLOCK_SIZE];

	for(int i = 0; i < BLOCK_SIZE; i++)
		for(int j = 0; j < BLOCK_SIZE; j++) {
			Block_A[i][j] = streamA.read();
			Block_B[i][j] = streamB.read();
	}	
	
	for(int i = 0; i < BLOCK_SIZE; i++) {
		for(int j = 0; j < BLOCK_SIZE; j++) {
			DTYPE val = 0;
			for(int k = 0; k < BLOCK_SIZE; k++)
				val += Block_A[i][k] * Block_B[k][j];
			streamC.write(val);
		}
	}
}

void blockmatmul(DTYPE *A, DTYPE *B, DTYPE *C) {
	#pragma HLS INTERFACE m_axi port=A bundle=gemm0 depth=N*M
	#pragma HLS INTERFACE m_axi port=B bundle=gemm1 depth=M*P
	#pragma HLS INTERFACE m_axi port=C bundle=gemm2 depth=N*P

    hls::stream<DTYPE> streamA("streamA");
    hls::stream<DTYPE> streamB("streamB");
    hls::stream<DTYPE> streamC("streamC");
	
	Block_i: for(int i = 0; i < N / BLOCK_SIZE; i++) 
		Block_j: for(int j = 0; j < P / BLOCK_SIZE; j++) 
			Block_k: for(int k = 0; k < M / BLOCK_SIZE; k++) {
				#pragma HLS dataflow
				load(A, streamA, i, k, M);
				load(B, streamB, k, j, P);
				compute(streamA, streamB, streamC);
				store(C, streamC, i, j, P);
			}
			
}

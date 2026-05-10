#include "block_mm.h"
#include <stdlib.h>
using namespace std;

void matmatmul_sw(DTYPE *A, DTYPE *B, DTYPE *out) {
	DTYPE sum = 0;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < P; j++){
			sum = 0;
			for(int k = 0; k < M; k++){
				sum += A[i * M + k] * B[k * P + j];
			}
			out[i * P + j] = sum;
		}
	}
}

int main() {
	
	DTYPE *A = (int *)malloc(N * M * sizeof(DTYPE));
	DTYPE *B = (int *)malloc(M * P * sizeof(DTYPE));
	DTYPE *matrix_swout = (int *)malloc(N * P * sizeof(DTYPE));
	DTYPE *matrix_hwout = (int *)malloc(N * P * sizeof(DTYPE));

	// init mat A
	for(int i = 0; i < N; i++)
 		for(int j = 0; j < M; j++) 
			A[i * N + j] = rand() % 512;

	// init mat B
	for(int i = 0; i < M; i++)
		for(int j = 0; j < P; j++)
			B[i * M + j] = rand() % 512;

	// clear output mat
	for(int i = 0; i < N; i++)
		for(int j = 0; j < P; j++) {
			matrix_swout[i * N + j] = 0;
			matrix_hwout[i * N + j] = 0;
		}

      		
	blockmatmul(A, B, matrix_hwout);

	matmatmul_sw(A, B, matrix_swout);

	bool fail = 0;
	for(int i = 0; i < N; i++)
		for(int j = 0; j < P; j++)
			if(matrix_swout[i * N + j] != matrix_hwout[i * N + j]) {
				printf("loc %d %d doesn't match, expected: %d, computed: %d\n", i, j, matrix_swout[i * N + j], matrix_hwout[i * N + j]); 
				return -1;
			} 

	cout << "passed" << endl;

	return 0;
}

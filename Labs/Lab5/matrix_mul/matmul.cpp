#include "matmul.h"

void matmul(int A[N][M], int B[M][P], int AB[N][P]) {	
	// on-chip RAMs, because the array size is already known in the compile time
	
	// for each row and column of AB 
	// which loop to pipeline?
	row: for(int i = 0; i < N; ++i) {
		col: for(int j = 0; j < P; ++j) {
			/* compute (AB)i,j */
			int ABij = 0;
			product: for(int k = 0; k < M; ++k) {
				ABij += A[i][k] * B[k][j];
			}
			AB[i][j] = ABij;
		}
	}
}


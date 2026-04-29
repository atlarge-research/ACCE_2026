#ifndef _BLOCK_MM_NO_DATAFLOW_H_
#define _BLOCK_MM_NO_DATAFLOW_H_
#include "hls_stream.h"
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

#define N 32
#define M 32
#define P 32

#define BLOCK_SIZE 4

typedef int DTYPE;

typedef struct { DTYPE a[BLOCK_SIZE]; } blockvec;

typedef struct { DTYPE out[BLOCK_SIZE][BLOCK_SIZE]; } blockmat;

void load(DTYPE *in, DTYPE Block[BLOCK_SIZE][BLOCK_SIZE], int ib, int jb, int row_size);

void store(DTYPE *out, DTYPE Block[BLOCK_SIZE][BLOCK_SIZE], int ib, int jb, int row_size);

void compute(DTYPE Block_A[BLOCK_SIZE][BLOCK_SIZE], DTYPE Block_B[BLOCK_SIZE][BLOCK_SIZE], DTYPE Block_C[BLOCK_SIZE][BLOCK_SIZE]);

void blockmatmul(DTYPE *A, DTYPE *B, DTYPE *C);
#endif

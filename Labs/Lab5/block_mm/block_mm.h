#ifndef _BLOCK_MM_H_
#define _BLOCK_MM_H_
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

void load(DTYPE *in, hls::stream<DTYPE> &stream, int ib, int jb, int row_size);

void store(DTYPE *out, hls::stream<DTYPE> &stream, int ib, int jb, int row_size);

void compute(hls::stream<DTYPE> &streamA, hls::stream<DTYPE> &streamB, hls::stream<DTYPE> &streamC);

void blockmatmul(DTYPE *A, DTYPE *B, DTYPE *C);
#endif

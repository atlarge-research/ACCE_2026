#include "hello_hls.h"

// HLS function to add two numbers
void add(int *a, int *b, int *c, int size) {
	/*
	#pragma HLS INTERFACE m_axi port=a depth=64
	#pragma HLS INTERFACE m_axi port=b depth=64
	#pragma HLS INTERFACE m_axi port=c depth=64

	#pragma HLS INTERFACE s_axilite port=size
	*/

	printf("%d\n", size);

	// set the size explicitly for latency estimation
	size = 64;
	
	// label the loop with a name, This will help you identify it in the synthesis report
    vec_add: for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

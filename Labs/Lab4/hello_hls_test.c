#include "hello_hls.h"
#include <assert.h>

int main() {
    int size = 64;
    int *a = (int *)malloc(size * sizeof(int));
    int *b = (int *)malloc(size * sizeof(int));
    int *c = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        a[i] = size * i;
        b[i] = size * i * i;
    }

    add(a, b, c, size);

	for (int i = 0; i < size; ++i) {
        printf("Hello, Vitis HLS! The sum of %d and %d is %d\n", a[i], b[i], c[i]);
		fflush(stdout);
		assert(c[i] == a[i] + b[i]);
	}

    free(a);
    free(b);
    free(c);

	return 0;
}

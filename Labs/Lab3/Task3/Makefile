# Makefile to compile all .cu files in the folder using nvcc

NVCC = module load cuda12.6/toolkit/12.6 && nvcc
SRCS = $(wildcard *.cu)
EXECS = $(SRCS:.cu=)

all: $(EXECS)

%: %.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(EXECS)

CC     ?= gcc
CFLAGS  = -O3 -march=native -std=gnu99 -ffp-contract=off -Wall -Wno-unknown-pragmas -Wno-unused-parameter
NVCC   ?= nvcc
ARCH   ?= sm_86          # change to your GPU, e.g. sm_75 (T4), sm_80 (A100), sm_89 (RTX 40xx)
ROUGH  ?= 0              # 0 = literal Eq.9, 1/2 = alternatives
DEFS    = -DROUGH_VARIANT=$(ROUGH)

all: bho_seq bho_omp
bho_seq: bho_seq.c bho_core.h bho_common.h
	$(CC) $(CFLAGS) $(DEFS) -o $@ $< -lm
bho_omp: bho_omp.c bho_core.h bho_common.h
	$(CC) $(CFLAGS) $(DEFS) -fopenmp -o $@ $< -lm
cuda: bho_cuda.cu bho_core.h bho_common.h
	$(NVCC) -O3 -arch=$(ARCH) --fmad=false $(DEFS) -o bho_cuda $<
clean:
	rm -f bho_seq bho_omp bho_cuda results_*.csv

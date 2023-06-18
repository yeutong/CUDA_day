# include <stdio.h>
extern "C" { // to prevent C++ name mangling, we use extern "C"
    __global__ void add_cuda_kernel(float *a, float *b, float *c, int size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size) {
            c[i] = a[i] + b[i];
        }
    }

    void add_cuda(float *a, float *b, float *c, int size, cudaStream_t *stream) {
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        add_cuda_kernel<<<gridDim, blockDim, 0, *stream>>>(a, b, c, size);
    }
}
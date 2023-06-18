# include <stdio.h>
extern "C" { // to prevent C++ name mangling, we use extern "C"
    __global__ void add3_cuda_kernel(float *a, float *b, float *c, float *d, int size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size) {
            d[i] = a[i] + b[i] + c[i];
        }
    }

    void add3_cuda(float *a, float *b, float *c, float *d, int size, cudaStream_t *stream) {
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        add3_cuda_kernel<<<gridDim, blockDim, 0, *stream>>>(a, b, c, d, size);
    }
}

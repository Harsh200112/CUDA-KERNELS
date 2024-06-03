#include<stdio.h>
#include<cuda.h>

__global__ void vectAdddition(float *A, float *B, float *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A, float *B, float *C, int n){
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);
    cudaMalloc((void**)&A_d, n);
    cudaMalloc((void**)&B_d, n);
    cudaMalloc((void**)&C_d, n);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, A, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 256;
    unsigned int numBlocks = (n + numThreads - 1)/numThreads;

    vectAdddition<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    const int n = 1000;
    
    float A[n], B[n], C[n];

    for (int i=0; i<n; i++){
        A[i] = float(i);
        B[i] = A[i] / 1000.0f;
    }

    vecAdd(A, B, C, n);

    for (int i=0; i<n; i++){
        printf("%8.3f", C[i]);
        printf(", ");
    }

    return 0;
}
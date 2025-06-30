#include <iostream>
#include <cstdlib>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void matmul_elem(int n, float *a, float *b, float *c){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < n && col < n){
        float dot_prod = 0.f;
        for(int i = 0; i < n; i++){
            dot_prod += a[row * n + i] * b[i * n + col];
        }
        c[row*n + col] = dot_prod;
    }

}

__global__ void matmult_elem_oneDim(int n, float *a, float *b, float *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / n;
    int col = idx % n;

    if(row < n && col < n){
        float dot_prod = 0.f;
        for(int i = 0; i < n; i++){
            dot_prod += a[row * n + i] * b[i * n + col];
        }
        c[row*n+col] = dot_prod;
    }
}

// One dimensional index when x, y z are given - 
// Z * Width * height + y * width + x

void setMatrix(int n, float * a, float *d_a){
    for(int i = 0; i < n * n; i++){
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    (cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice));
}

int main(){
    int n = 200;
    float *a, *b, *c, *d_a, *d_b, *d_c;
    
    a = new float[n * n];
    b = new float[n * n];
    c = new float[n * n];

    (cudaMalloc(&d_a, n * n * sizeof(float)));
    (cudaMalloc(&d_b, n * n * sizeof(float)));
    (cudaMalloc(&d_c, n * n * sizeof(float)));

    setMatrix(n, a, d_a);
    setMatrix(n, b, d_b);

    int totalThreads = n * n;
    int threadsPerBlock = 512;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    matmult_elem_oneDim<<<blocks, threadsPerBlock>>>(n, d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for(int i = 0 ; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << c[i * n + j] << " ";
        }
        std::cout<<std::endl;
    }

    return 0;
}
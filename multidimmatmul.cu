// One dimensional index when x, y z are given - 
// Z * Width * height + y * width + x

/*
A three dim array a, 2 dim array b, 1 dim array c
output three dim array like this
out[x][y][z] = a[x][y][z] + b[x][y] + c[x]
*/

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

__global__ void StreamedOut(int n, float *a, float *b, float *c, float *out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n * n * n){
        int l = i / (n * n);
        int h = (i / n) % n;
        int w = i % n;
        out[i] = a[i] + b[h * n + w] + c[l];
    }
} 


void print(int n, float *a){
    for(int k = 0; k < n; k++){
        std::cout<<"Layer "<< k << std::endl;

        for(int i = 0 ; i < n; i++){
            for(int j = 0; j < n; j++){
                std::cout << a[(k * n * n + i * n + j)] << " ";
            }
            std::cout<<std::endl;
        }
    }
}

int main(){
    int n = 20;
    float *out, *out_d, *a, *a_d, *b, *b_d, *c, *c_d;
    out = new float[n * n * n];
    a = new float[n * n * n];
    b = new float[n * n];
    c = new float[n];

    for(int i = 0; i < n * n * n; i++) a[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < n * n; i++) b[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < n; i++)c[i] = static_cast<float>(rand()) / RAND_MAX; 

    cudaCheckError(cudaMalloc(&out_d, n * n * n * sizeof(float)));
    cudaCheckError(cudaMalloc(&a_d, n * n * n * sizeof(float)));
    cudaCheckError(cudaMalloc(&b_d, n * n * sizeof(float)));
    cudaCheckError(cudaMalloc(&c_d, n * sizeof(float)));
    
    cudaCheckError(cudaMemcpy(a_d, a, n * n * n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(b_d, b, n * n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(c_d, c, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = n * n * n;
    int threadsPerBlock = 512 ? (n > 512) : 256;
    int blocks = (threads + threadsPerBlock - 1) / threadsPerBlock;

    StreamedOut<<<blocks, threadsPerBlock>>>(n, a_d, b_d, c_d, out_d);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(out, out_d, n * n * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(out_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    print(n, out);
}   
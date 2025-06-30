#include <iostream>
#include <fstream>
__global__ void helloWorld(int* thread_id, int* block_id, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        // printf("Hello World from thread %d, block %d\n", threadIdx.x, blockIdx.x);
        thread_id[i] = threadIdx.x;
        block_id[i] = blockIdx.x;
    }
}

int main() {
    int n = 1024 * 128;
    int threadsPerBlock = 1024;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    int *thread_id, *d_thread_id, *block_id, *d_block_id;
    thread_id = (int*)malloc(n * sizeof(int));
    block_id = (int*)malloc(n * sizeof(int));
    
    cudaMalloc(&d_thread_id, n * sizeof(int));
    cudaMalloc(&d_block_id, n * sizeof(int));

    helloWorld<<<blocks, threadsPerBlock>>>(d_thread_id, d_block_id, n);
    cudaDeviceSynchronize();

    cudaMemcpy(thread_id, d_thread_id, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_id, d_block_id, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream file("hellowWorldOutput.txt");
    for(int i = 0; i < n; i++){
        file << "Thread " << thread_id[i] << " in block " << block_id[i] << std::endl;
    }
    file.close();
    cudaFree(d_thread_id);
    cudaFree(d_block_id);
    free(thread_id);
    free(block_id);
    return 0;
}
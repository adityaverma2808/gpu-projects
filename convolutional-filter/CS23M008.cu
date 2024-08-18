#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

// CUDA kernel for convolution
__global__ void dkernel(long int *ip, long int *op, long int *g_filter, int k, int row, int col)
{
    extern __shared__ long int filter[];

    // Load the entire filter into shared memory
    unsigned long int id = threadIdx.y * blockDim.x + threadIdx.x;
    for(;id < k * k; id += blockDim.x * blockDim.y)
    {
        filter[id] = g_filter[id];
    }
    __syncthreads();

    // Compute global indices
    int ii = blockIdx.y * blockDim.y + threadIdx.y;
    int jj = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute convolution for each element
    bool isCorrectId = ii < row && jj < col;
    if (isCorrectId)
    {
        int half_k = k / 2;
        long int total = 0;
        int i = 0;
        while(i < k*k){
            int a = i / k;
            int b = i % k;
            int row_r = ii + a - half_k;
            int col_r = jj + b - half_k;
            bool flag = col_r >= 0 && row_r >= 0 && row_r < row && col_r < col;
            if (flag)
            {
                total = total + (ip[row_r * col + col_r] * filter[a * k + b]);
            }
            i++;
        }
        op[ii * col + jj] = total;
    }
}


int main() {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int ii = 0; ii < m * n; ii++) {
        cin>>h_mat[ii];
    }

    for (long int ii = 0; ii < k * k; ii++) {
        cin>>h_filter[ii];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    /*Start Here*/
    long int *gpu_mat;
    long int *gpu_filter;
    long int *gpu_output;

    cudaMalloc(&gpu_mat, m*n*sizeof(long int));
    cudaMemcpy(gpu_mat, h_mat, m*n*sizeof(long int), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_output, m*n*sizeof(long int));

    cudaMalloc(&gpu_filter, k*k*sizeof(long int));
    cudaMemcpy(gpu_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

   // dkernel<<<m, n, k*k*sizeof(long int)>>>(gpu_mat, gpu_output, gpu_filter, k, m, n);
   int blockSizeX = 16;  // Choose an appropriate block size (e.g., 16x16)
    int blockSizeY = 16;
    dim3 blockSize(blockSizeX, blockSizeY);

    // Compute grid dimensions
    int gridSizeX = (n + blockSizeX - 1) / blockSizeX;  // Grid size along x-axis
    int gridSizeY = (m + blockSizeY - 1) / blockSizeY;  // Grid size along y-axis
    dim3 gridSize(gridSizeX, gridSizeY);

    // Shared memory size for filter
    int sharedMemSize = k * k * sizeof(long int);

    // Launch kernel
    dkernel<<<gridSize, blockSize, sharedMemSize>>>(gpu_mat, gpu_output, gpu_filter, k, m, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch

    cudaMemcpy(h_ans, gpu_output, m*n*sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_mat);
    cudaFree(gpu_filter);
    cudaFree(gpu_output);
    //$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
  */
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }
    return 0;
}
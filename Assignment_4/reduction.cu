//This is a tutorial program for COMP5112 - assignment 4

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <time.h>
#include <getopt.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::ceil;
using std::memcpy;

/*
 * This is a CHECK function to check CUDA calls
 */
#define CHECK(call)                                                            \
 {                                                                              \
     const cudaError_t error = call;                                            \
     if (error != cudaSuccess)                                                  \
     {                                                                          \
         fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
         fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                 cudaGetErrorString(error));                                    \
         exit(1);                                                               \
     }                                                                          \
 }

__global__ void reduction1(int *d_input_data, int *block_local_sum, int N){
    int element_per_thread = (int) ceil(N * 1.0 / (blockDim.x * gridDim.x));

    //store the sum for each thread
    __shared__ int thread_local_sum[1024];

    int my_start = (blockIdx.x * blockDim.x + threadIdx.x) * element_per_thread;
    int my_end = my_start + element_per_thread;
    if(my_end >= N){
        my_end = N;
    }

    int my_sum = 0;
    for(int i = my_start ; i < my_end; i++){
        my_sum += d_input_data[i];
    }
    //store my result to shared memory
    thread_local_sum[threadIdx.x] = my_sum;
    __syncthreads();//synchronization, make sure every threads has done their summation

    //use the first thread in this block to calculate the local summation of this block
    if(threadIdx.x == 0){
        int this_block_sum = 0;
        for (int i = 0 ; i < blockDim.x; i++){
            this_block_sum += thread_local_sum[i];
        }

        //store this block summation to global memory address
        block_local_sum[blockIdx.x] = this_block_sum;
    }
}

__global__ void reduction2(int *block_local_sum, int *d_sum){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        int all_sum = 0;
        //sum all block's result up to get the global summation
        for(int i = 0; i < gridDim.x; i++) {
            all_sum += block_local_sum[i];
        }

        d_sum[0] = all_sum;
    }
}

int main(int argc, char **argv){

    assert(argc > 1);
    string filename = argv[1];
    cudaDeviceReset();

    int N; // size of input
    int *h_input_data;
    int *d_input_data, *d_block_local_sum;
    int *h_sum, *d_sum;

    std::ifstream inputf(filename, std::ifstream::in);
    inputf >> N;


    //kernel configuration
    dim3 blocks(8);
    dim3 threads(1024);

    //memory allocation
    h_input_data = (int *) malloc(sizeof(int) * N);
    h_sum = (int*) malloc(sizeof(int));
    cudaMalloc(&d_input_data, sizeof(int) * N);
    cudaMalloc(&d_block_local_sum, sizeof(int) * 8);
    cudaMalloc(&d_sum, sizeof(int));

    int host_result = 0;
    //read input file
    for(int i = 0 ; i < N; i++){
        inputf >> h_input_data[i];
        host_result += h_input_data[i];
    }

    //copy input data to device
    cudaMemcpy(d_input_data, h_input_data, sizeof(int) * N, cudaMemcpyHostToDevice);

    //time counter
    timeval start_wall_time_t, end_wall_time_t;
    float ms_wall;
    //start timer
    gettimeofday(&start_wall_time_t, nullptr);

    reduction1 <<<blocks, threads>>> (d_input_data, d_block_local_sum, N);
    reduction2 <<<blocks, threads>>> (d_block_local_sum, d_sum);
    CHECK(cudaDeviceSynchronize());
    //end timer

    gettimeofday(&end_wall_time_t, nullptr);
    ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
               + end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

    std::cerr << "Time(ms): " << ms_wall << endl;

    //copy result data back
    cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Host result: " << host_result << std::endl;
    std::cout << "Device result: " << h_sum[0] << std::endl;


    free(h_input_data);
    free(h_sum);
    cudaFree(d_block_local_sum);
    cudaFree(d_input_data);
    cudaFree(d_sum);

    return 0;

}
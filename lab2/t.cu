#include <stdio.h>
#include <cuda.h>

#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

__device__ int getGlobalIdx_1Dg_1Db(){
  return blockIdx.x*blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1Dg_2Db(){
  return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2Dg_1Db(){
  return get_bid() * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2Dg_2Db(){
  return get_bid()*(blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
}


int main(int argc, char **argv){      
  int N = 100;
  dim3 block(3);
  dim3 grid( (N+block.x-1) / block.x);

}
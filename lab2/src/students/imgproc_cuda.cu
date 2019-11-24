// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include "imgproc_cuda.hpp"
#include <typeinfo>

double CUDATime[4];


static void checkSuccess(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

__device__ 
unsigned int getGlobalIdx_1Dg_1Db_X(){
  return blockIdx.x*blockDim.x + threadIdx.x;
}

__device__ 
unsigned int getGlobalIdx_1Dg_1Db_Y(){
  return blockIdx.y*blockDim.y + threadIdx.y;
}

__device__ 
unsigned int getGlobalIdx_1Dg_2Db(){
  return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ 
void check(void)
{
    printf("threadIdx:(%d, %d)\n", threadIdx.x, threadIdx.y);
    printf("blockIdx:(%d, %d)\n", blockIdx.x, blockIdx.y);
    printf("blockDim:(%d, %d)\n", blockDim.x, blockDim.y);
    printf("gridDim:(%d, %d)\n", gridDim.x, gridDim.y);
}

static inline void checkDimensionsEqualOrThrow(const Image *a, const Image *b) {
  assert(a != nullptr);
  assert(b != nullptr);
  if ((a->width != b->width) || (a->height != b->height)) {
    throw std::domain_error("Source and destination image are not of equal dimensions.");
  }
}

static inline void checkValidColorChannelOrThrow(int channel) {
  if ((channel < 0) || (channel > 3)) {
    throw std::domain_error("Color channel must be 0,1,2 or 3.");
  }
}

__global__
void ConvoluteCUDAKernel_1(unsigned char* raw, unsigned char* dest, float* kernel_weights, unsigned int width, unsigned int height, int kWidth, int kHeight,float kScale){
  
  int x = getGlobalIdx_1Dg_1Db_X();
  int y = getGlobalIdx_1Dg_1Db_Y();
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y;
  
  
  //__shared__ float kk[121];
  
  
  for (int j = y; j < height; j+=step_y) {
    for (int i = x; i < width; i+=step_x) {
        for(int m=0;m<4;m++){
          //__shared__ float cc;
          float cc = 0.0;
          for (int kj = -kHeight / 2; kj <= kHeight / 2; kj++) {
            for (int ki = -kWidth / 2; ki <= kWidth / 2; ki++) {
              int ci = i + ki;
              int cj = j + kj;
              if ((ci >= 0) && (cj >= 0) && (ci < width) && (cj < height)) {
                auto v = (float) raw[4 * (cj * width + ci) + m];
                auto k = kernel_weights[(kj + kHeight / 2) * kWidth + (ki + kWidth / 2)];
                cc += v * k;
              }
            }
          }
          dest[4 * (j * width + i) + m] = (unsigned char) (cc * kScale);
        }
    }
  }  
  
}

void convoluteCUDA(const Image *src, Image *dest, const Kernel *kernel){
  assert((src != nullptr) && (dest != nullptr) && (kernel != nullptr));
  checkDimensionsEqualOrThrow(src, dest);
  
  unsigned int imgSize = src->width * src->height;
  unsigned int kernelSize = kernel->width * kernel->height;
  unsigned char* input = NULL;
  unsigned char* output = NULL;
  float* kernelW;
  
  checkSuccess(cudaMalloc((void**)&input,imgSize*4*sizeof(unsigned char)));
  checkSuccess(cudaMalloc((void**)&output,imgSize*4*sizeof(unsigned char)));
  checkSuccess(cudaMalloc((void**)&kernelW,kernelSize*sizeof(float)));
  
  checkSuccess(cudaMemcpy(input,src->raw.data(),imgSize*4*sizeof(unsigned char),cudaMemcpyHostToDevice));
  checkSuccess(cudaMemcpy(kernelW,kernel->weights.data(),kernelSize*sizeof(float),cudaMemcpyHostToDevice));
  
  dim3 block(32,32);
  dim3 grid((src->width+31)/32,(src->height+31)/32);
  
  ConvoluteCUDAKernel_1<<<grid,block>>>(input,output,kernelW,src->width,src->height,kernel->width,kernel->height,kernel->scale);
  
  checkSuccess(cudaMemcpy(dest->raw.data(),output,imgSize*4*sizeof(unsigned char),cudaMemcpyDeviceToHost));
  
  cudaFree((void**)input);
  cudaFree((void**)output);
  
}

__global__ 
void HistogramCudaKernel_1(Pixel* pixel, int* output, unsigned int width, unsigned int height){

  __shared__ unsigned int temp[1024];
  
  int tx = threadIdx.x;
  if(tx<1024) temp[tx] = 0;
  __syncthreads();
  
  int x = getGlobalIdx_1Dg_1Db_X();
  int step_x = blockDim.x * gridDim.x;
  while (x < width*height)
  {
    auto i0 = pixel[x].colors[0];
    auto i1 = pixel[x].colors[1];
    auto i2 = pixel[x].colors[2];
    auto i3 = pixel[x].colors[3];
    atomicAdd( &temp[0*256+i0], 1);
    atomicAdd( &temp[1*256+i1], 1);
    atomicAdd( &temp[2*256+i2], 1);
    atomicAdd( &temp[3*256+i3], 1);
    x += step_x;
  }
  __syncthreads();
  
  atomicAdd( &(output[threadIdx.x]), temp[threadIdx.x] );
  
}

__global__
void HistogramCudaKernel_4(unsigned char* input, int* output, unsigned int width, unsigned int height){
  
  int x = threadIdx.x;
  int y = blockIdx.x;
  int stride = blockDim.x;
  
  //__shared__ unsigned int temp[1024*4];
  //temp[threadIdx.x] = 0;
  //__syncthreads();
  
  for(int m=0;m<4;m++){
        auto intencity = input[4 * (y * stride + x) + m];
        atomicAdd(&output[256 * m + intencity],1);
      }
  
  //__syncthreads();
  //atomicAdd(&(output[threadIdx.x]),temp[threadIdx.x]);

}

Histogram getHistogramCUDA(const Image *src){

  assert((src != nullptr));
  
  Histogram hist;
  
  unsigned int imgSize = src->height*src->width;
  Pixel* input = NULL;
  int* output;
  
  printf("------%d\n",src->height);
  printf("------%d\n",src->width);
  
  checkSuccess(cudaMalloc((void**)&input,imgSize*4*sizeof(unsigned char)));
  checkSuccess(cudaMalloc(&output,256*4*sizeof(int)));
  checkSuccess(cudaMemcpy(input,src->pixels,imgSize*4*sizeof(unsigned char),cudaMemcpyHostToDevice));
  
  int block = 1024;
  int grid = (imgSize + block -1)/block;
  
  HistogramCudaKernel_1<<<grid,block>>>(input,output,src->width,src->height);
  
  cudaMemcpy(hist.values.data(), output, 256*4*sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree((void*)input);
  cudaFree((void*)output);
  
  return hist;
  
}

Histogram getHistogramC(const Image *src) {
  assert((src != nullptr));
  Histogram hist;
  for (int y = 0; y < src->height; y++) {
    for (int x = 0; x < src->width; x++) {
      for (int c = 0; c < 4; c++) {
        auto intensity = src->pixel(x, y).colors[c];
        hist(intensity, c)++;
      }
    }
  }
  return hist;
}


__global__
void enhanceContrastLinearlyCudaKernel_1(unsigned char* raw, unsigned char* dest, unsigned int width, unsigned int height, unsigned char f1, unsigned char l1, float s1 ,unsigned char f2, unsigned char l2, float s2,unsigned char f3, unsigned char l3, float s3){

  int x = getGlobalIdx_1Dg_1Db_X(); 
  int y = getGlobalIdx_1Dg_1Db_Y(); 
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y; 
  
  unsigned char first[3] = {f1,f2,f3}; 
  unsigned char last[3] = {l1,l2,l3}; 
  float scale[3] = {s1,s2,s3}; 
  
  for (int i = x; i < width; i += step_x) {
    for (int j = y; j < height; j += step_y) {
        for(int m=0;m<3;m++){
          if (raw[4 * (j * width + i) + m] < first[m]) {
            dest[4 * (j * width + i) + m] = 0;
          } else if (raw[4 * (j * width + i) + m] > last[m]) {
            dest[4 * (j * width + i) + m] = 255;
          } else {
            unsigned char t = (unsigned char) (scale[m] * (raw[4 * (j * width + i) + m] - first[m]));
            dest[4 * (j * width + i) + m] = t;
          }
        }
        dest[4 * (j * width + i) + 3] = raw[4 * (j * width + i) + 3];
    }
  }

}


void enhanceContrastLinearlyCUDA(const Image *src, const Histogram *src_hist, Image *dest, int low, int high){
  assert((src != nullptr) && (src_hist != nullptr) && (dest != nullptr));
  checkDimensionsEqualOrThrow(src, dest);
  
  unsigned char f[3] = {0,0,0};
  unsigned char l[3] = {255,255,255};
  float s[3] = {0,0,0};
  
  for(int i=0;i<3;i++)
    while( (f[i] < src_hist->range) && src_hist->count(f[i], i) <= low)
      f[i]++;
  
  for(int i=0;i<3;i++)
    while( (l[i] > f[i]) && src_hist->count(l[i], i) <= high)
      l[i]--;
  
  for(int i=0;i<3;i++)
    s[i] = 255.0f / (l[i] - f[i]);
    
  
  unsigned int imgSize = src->width * src->height;
  unsigned char* input = NULL;
  unsigned char* output = NULL;
   
  checkSuccess(cudaMalloc((void**)&input,imgSize*4*sizeof(unsigned char)));
  checkSuccess(cudaMalloc((void**)&output,imgSize*4*sizeof(unsigned char)));
  
  checkSuccess(cudaMemcpy(input,src->raw.data(),imgSize*4*sizeof(unsigned char),cudaMemcpyHostToDevice));
  
  int blockSize = 32;
  int gridSize = (imgSize + blockSize - 1)/blockSize;
  dim3 dimBlock(blockSize,blockSize);
  dim3 dimGrid(gridSize,gridSize);
  
  enhanceContrastLinearlyCudaKernel_1<<<gridSize,blockSize>>>(input,output,src->width,src->height,f[0],l[0],s[0],f[1],l[1],s[1],f[2],l[2],s[2]);
  
  
  checkSuccess(cudaMemcpy(dest->raw.data(),output,imgSize*4*sizeof(unsigned char),cudaMemcpyDeviceToHost));
  
  cudaFree((void**)input);
  cudaFree((void**)output);
  
}

__global__
void applyRippleCUDAKernel_1(unsigned char* raw, unsigned char* dest, unsigned int width, unsigned int height, float frequency){
  
  int x = getGlobalIdx_1Dg_1Db_X();
  int y = getGlobalIdx_1Dg_1Db_Y();
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y;
  
  for (int j = y; j < height; j+=step_y) {
    for (int i = x; i < width; i+=step_x) {
    
      //dest[0]=dest[1]=dest[2]=dest[3]=0;
      float ni = -1.0f + (2.0f * i) / width;
      float nj = -1.0f + (2.0f * j) / height;
      auto dist = sqrt(pow(nj, 2) + pow(ni, 2));
      float angle = atan2f(nj, ni);
      auto src_dist = pow(sin(dist * M_PI / 2.0 * frequency), 2);

      if ((src_dist > 1.0f)) {
        continue;
      }

      auto nsi = src_dist * cos(angle);
      auto nsj = src_dist * sin(angle);
      auto si = int((nsi + 1.0) / 2 * width);
      auto sj = int((nsj + 1.0) / 2 * height);
      if ((si >= width) || (sj >= height)) {
        continue;
      }
      for(int m=0;m<4;m++)
        dest[4 * (j * width + i) + m] = raw[4 * (sj * width + si) + m];
    }
  }
}

void applyRippleCUDA(const Image *src, Image *dest, float frequency){
  
  assert((src != nullptr) && (dest != nullptr));
  checkDimensionsEqualOrThrow(src, dest);
  
  unsigned int imgSize = src->width * src->height;
  unsigned char* input = NULL;
  unsigned char* output = NULL;
  
  checkSuccess(cudaMalloc((void**)&input,imgSize*4*sizeof(unsigned char)));
  checkSuccess(cudaMalloc((void**)&output,imgSize*4*sizeof(unsigned char)));
  
  checkSuccess(cudaMemcpy(input,src->raw.data(),imgSize*4*sizeof(unsigned char),cudaMemcpyHostToDevice));
  
  int blockSize = 32;
  dim3 block(blockSize,blockSize);
  int h = (src->width+blockSize-1)/blockSize;
  int w = (src->height+blockSize-1)/blockSize;
  dim3 grid(h,w); 
  
  applyRippleCUDAKernel_1<<<grid,block>>>(input,output,src->width,src->height,frequency);
  
  checkSuccess(cudaMemcpy(dest->raw.data(),output,imgSize*4*sizeof(unsigned char),cudaMemcpyDeviceToHost));
  
  cudaFree((void**)input);
  cudaFree((void**)output);
  
}

void copyChannelCUDA(const Image *src, Image *dest, int channel){
  
  assert((src != nullptr) && (dest != nullptr));
  checkDimensionsEqualOrThrow(src, dest);
  checkValidColorChannelOrThrow(channel);
  
  for (int y = 0; y < src->height; y++) {
    for (int x = 0; x < src->width; x++) {
      dest->pixel(x, y).colors[channel] = src->pixel(x, y).colors[channel];
    }
  }

}


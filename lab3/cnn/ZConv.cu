#include "stdio.h"
#include "ZConv.hpp"
#include "tensor_t.h"
#include <iostream>

#define LEARNING_RATE 0.01f
#define MOMENTUM 0.6f
#define WEIGHT_DECAY 0.001f

#define blockSize 32

conv_layer_t::conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size):
      layer_t(layer_type::conv,
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(
                  (in_size.x - extend_filter) / stride + 1,
                  (in_size.y - extend_filter) / stride + 1,
                  number_filters)
      ), stride(stride), extend_filter(extend_filter)
  {
    assert((float(in_size.x - extend_filter) / stride + 1) == ((in_size.x - extend_filter) / stride + 1));
    assert((float(in_size.y - extend_filter) / stride + 1) == ((in_size.y - extend_filter) / stride + 1));

    for (int a = 0; a < number_filters; a++) {
      tensor_t<float> t(extend_filter, extend_filter, in_size.z);

      int maxval = extend_filter * extend_filter * in_size.z;

      for (int i = 0; i < extend_filter; i++)
        for (int j = 0; j < extend_filter; j++)
          for (int z = 0; z < in_size.z; z++)
            t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
      filters.push_back(t);
    }
    for (int i = 0; i < number_filters; i++) {
      tensor_t<gradient_t> t(extend_filter, extend_filter, in_size.z);
      filter_grads.push_back(t);
    }

  fSize = filters.size();
  filterSize = sizeof(float)*this->extend_filter*this->extend_filter*inSizeZ;
  fullFilterSize = filterSize*fSize;
  outputSize *= fSize;
  
  //gX = this->extend_filter;
  //gY = this->extend_filter;
  //gZ = inSizeZ;
  //gradSize = sizeof(float)*gX*gY*gZ*this->filter_grads.size();
  
  cudaMalloc(&this->input,this->inputSize);
  cudaMalloc(&this->output,this->outputSize);
  cudaMalloc(&this->filterData,this->fullFilterSize);
  cudaMalloc(&this->oldGrads,this->fullFilterSize);
  
    cudaMalloc(&gg,800); // 5* 5* 8* 4
  cudaMalloc(&nextGrads,24*24*8*4);
  
}

__global__ 
void activateCUDA(
  float* input, 
  float* output,
  int inSizeX,
  int inSizeY,
  int fSize, 
  float* filter_data, 
  uint16_t extend_filter, 
  uint16_t stride, 
  int inSizeZ, int outSizeX, int outSizeY){ 
  
  int cuda_x = blockIdx.x * blockDim.x + threadIdx.x; 
  int cuda_y = blockIdx.y * blockDim.y + threadIdx.y; 
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y; 
  
  for (int filter = 0; filter < fSize; filter++) {
      for (int x = cuda_x; x < outSizeX; x += step_x) {
        for (int y = cuda_y; y < outSizeY; y += step_y) {
          point_t mapped;
          mapped.x = (uint16_t)x * stride;
          mapped.y = (uint16_t)y * stride;
          mapped.z = 0;
          float sum = 0;
          //printf("----%f\n",sum);
          
            for (int i = 0; i < extend_filter; i++)
              for (int j = 0; j < extend_filter; j++){
                for (int z = 0; z < inSizeZ; z++) {
                  float v = input[z * (inSizeX * inSizeY) + (mapped.y + j) * (inSizeX) + mapped.x + i];
                  float f = filter_data[filter * (extend_filter * extend_filter) + j * (extend_filter) + i];
                  sum += f * v;
                  //printf("----%f\n",sum);
                }
                //__syncthreads();
              }
          output[filter*(outSizeX*outSizeY) + y * (outSizeX) + x] = sum;
          //output[ y * (outSizeX) + x] = sum;
        }
      }
    }
}

void conv_layer_t::activate(tensor_t<float> &in)
{
  this->in = in;

  float fullFilters[this->fullFilterSize/sizeof(float)];
  for(int i=0;i<this->fSize;i++){
    for(int j=0;j<this->filters[i].data.size();j++){
      fullFilters[i*this->filters[i].data.size()+j] = this->filters[i].data[j];    
    }
  }
  
  cudaMemcpy(this->filterData,fullFilters,this->fullFilterSize,cudaMemcpyHostToDevice);
  cudaMemcpy(this->input,in.data.data(),this->inputSize,cudaMemcpyHostToDevice);
  dim3 block(blockSize,blockSize);
  dim3 grid((this->in.size.x+blockSize-1)/blockSize,(this->in.size.y+blockSize-1)/blockSize);

  activateCUDA<<<grid,block>>>(
    this->input,
    this->output,
    in.size.x,
    in.size.y,
    this->fSize,
    this->filterData,
    this->extend_filter,
    this->stride,
    this->inSizeZ,
    this->outSizeX,
    this->outSizeY);
    
  cudaMemcpy(this->out.data.data(),this->output,this->outputSize,cudaMemcpyDeviceToHost);
  //cudaMemcpy(&this->out.data[i*(outSizeX*outSizeY)],this->output,this->outputSize,cudaMemcpyDeviceToHost);

}


void conv_layer_t::fix_weights()
{
    for (int a = 0; a < filters.size(); a++)
      for (int i = 0; i < extend_filter; i++)
        for (int j = 0; j < extend_filter; j++)
          for (int z = 0; z < in.size.z; z++) {
            float &w = filters[a].get(i, j, z);
            // this for the gradients calculation
            gradient_t &grad = filter_grads[a].get(i, j, z);
            // weigth update in optimisation method
            w = update_weight(w, grad);
            // gradient update in optimisation method
            update_gradient(grad);
          }
  }

__device__ int normalize_range(float f, int max, bool lim_min) {
    if (f <= 0)
      return 0;
    max -= 1;
    if (f >= max)
      return max;

    if (lim_min) // left side of inequality
      return (int)(ceil((double)f));
    else
      return (int)(floor((double)f));
  }
  
struct range_t {
    int min_x, min_y, min_z;
    int max_x, max_y, max_z;
};

  // calculate the output position during convolution while know input position
__device__ range_t mapCUDA(int x, int y, uint16_t extend_filter, uint16_t stride, int outSizeX,int outSizeY, int fSize) {
    float a = x;
    float b = y;
    return
        {
            normalize_range((a - extend_filter + 1) / stride, outSizeX, true),
            normalize_range((b - extend_filter + 1) / stride, outSizeY, true),
            0,
            normalize_range(a / stride, outSizeX, false),
            normalize_range(b / stride, outSizeY, false),
            (int) fSize - 1,
        };
  }
  
  



__global__
void calcCUDA(
  float* input,
  float* nextGrads,
  float* grads,
  uint16_t extend_filter,
  uint16_t stride,
  int outSizeX,
  int outSizeY,
  int fSize,
  int inSizeX,
  int inSizeY,
  int inSizeZ
  ){
  int cuda_x = blockIdx.x * blockDim.x + threadIdx.x;
  int cuda_y = blockIdx.y * blockDim.y + threadIdx.y;
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y;
  
  
  for (int x = cuda_x; x < inSizeX; x += step_x) {
      for (int y = cuda_y; y < inSizeY; y += step_y) {
        int minX = (x - extend_filter + 1)<0?0:(x - extend_filter + 1);
        int minY = (y - extend_filter + 1)<0?0:(y - extend_filter + 1);
        int minZ = 0;
        int maxX = x>=23?23:x;
        int maxY = y>=23?23:y;
        int maxZ = (int) fSize - 1;
        for (int z = 0; z < inSizeZ; z++) {
          for (int i = minX; i <= maxX; i++) {
            int minx = i * stride;
            for (int j = minY; j <= maxY; j++) {
              int miny = j * stride;
              for (int k = minZ; k <= maxZ; k++) {
                float v1 = input[y*(28)+x];
                float v2 = nextGrads[k*(24*24)+j*(24)+i];
                float value = v1 * v2;
                atomicAdd(&grads[(k+z)*(extend_filter*extend_filter)+(y-miny)*(extend_filter)+(x-minx)],value);
              }
            }
          }
        }
      }
    }
}
  
void conv_layer_t::calc_grads(tensor_t<float> &grad_next_layer)
{

  for (int k = 0; k < filter_grads.size(); k++) {
      for (int i = 0; i < extend_filter; i++)
        for (int j = 0; j < extend_filter; j++)
          for (int z = 0; z < in.size.z; z++){
            filter_grads[k].get(i, j, z).grad = 0;
            }
    }
  
  //std::cout<< "OK" << std::endl;
  

  float zeroF[8*25];
  
  int nextSize = sizeof(float)*grad_next_layer.data.size(); //24 * 24 * 8  *4
  
  cudaMemcpy(nextGrads,grad_next_layer.data.data(),nextSize,cudaMemcpyHostToDevice);
  
  memset(zeroF,0,sizeof(zeroF));
  cudaMemcpy(gg,zeroF,800,cudaMemcpyHostToDevice);

  dim3 block(blockSize,blockSize);
  dim3 grid((this->in.size.x+blockSize-1)/blockSize,(this->in.size.y+blockSize-1)/blockSize);
  
  calcCUDA<<<grid,block>>>(
    this->input, 
    this->nextGrads,
    this->gg,
    extend_filter,
    stride,
    this->outSizeX,
    this->outSizeY,
    this->fSize,
    this->in.size.x,
    this->in.size.y,
    this->inSizeZ);
    
  cudaMemcpy(zeroF,gg,this->fullFilterSize,cudaMemcpyDeviceToHost);
    
  for (int k = 0; k < filter_grads.size(); k++) 
      for (int i = 0; i < extend_filter; i++)
        for (int j = 0; j < extend_filter; j++)
          for (int z = 0; z < in.size.z; z++){
            filter_grads[k].get(i, j, z).grad = zeroF[k*(extend_filter*extend_filter)+j*extend_filter+i];
            }
  
}

/*
*/


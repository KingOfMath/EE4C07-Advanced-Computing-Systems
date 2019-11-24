#include "stdio.h"
#include <iostream>
#include "ZFc.hpp"
#include "tensor_t.h"

#define LEARNING_RATE 0.01f
#define MOMENTUM 0.6f
#define WEIGHT_DECAY 0.001f


fc_layer_t::fc_layer_t(tdsize in_size, int out_size):
      layer_t(layer_type::fc,
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(out_size, 1, 1)), weights(in_size.x * in_size.y * in_size.z, out_size, 1) {
    input = std::vector<float>(static_cast<unsigned long>(out_size));
    gradients = std::vector<gradient_t>(static_cast<unsigned long>(out_size));

    int maxval = in_size.x * in_size.y * in_size.z;

    // weight initialization
    for (int i = 0; i < out_size; i++)
      for (int h = 0; h < in_size.x * in_size.y * in_size.z; h++)
        weights(h, i, 0) = 2.19722f / maxval * rand() / float(RAND_MAX);
    // 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
    
    cudaMalloc(&Weight,weightSize);
    cudaMalloc(&inData,sizeof(float)*input.size());
    cudaMalloc(&temp,nSize);
    cudaMalloc(&outData,nSize);
    
    cudaMalloc((void**)&inImage,inputSize);
    
  }
  
__device__  float activator_function(float x) {
    //return tanhf( x );
    float sig = 1.0f / (1.0f + exp(-x));
    return sig;
  }

__device__  float activator_derivativeCUDA(float x) {
    //float t = tanhf( x );
    //return 1 - t * t;
    float sig = 1.0f / (1.0f + exp(-x));
    return sig * (1 - sig);
  }
  
__device__
int mapping(point_t d, int sizex, int sizey) {
  return d.z * (sizex * sizey) +
      d.y * (sizex) +
      d.x;
}  
  
  

__global__
void activatefc1(int sizez,
                int sizex, 
                int sizey, 
                int sizewx, 
                float* weights,
                float* input,                
                float* outData){ 

  int cuda_x =  threadIdx.x ;
  float sum = 0.0;
  
    for (int i = 0; i < sizex; i++)
        for (int j = 0; j < sizey; j++) 
          for (int z = 0; z < sizez; z++) {//z=8
                    
            int m = mapping({i, j, z},sizex,sizey);
            sum += input[z*(sizex*sizey) + j*sizex + i]* weights[cuda_x*sizewx/10+m];
            
          }
    outData[cuda_x] = sum;
 
  }


void fc_layer_t :: activate(tensor_t<float> &in) {
  
 // int blockSize = 8;
  this->in = in;
  
  
  cudaMemcpy(inImage,this->in.data.data(),inputSize,cudaMemcpyHostToDevice);
  cudaMemcpy(Weight,this->weights.data.data(),weightSize,cudaMemcpyHostToDevice);
  
  int block=outSizeX;
  int grid =1;

  activatefc1<<<grid,block>>>(
  in.size.z,
  in.size.x,
  in.size.y,
  this->weightSize/sizeof(float), 
  Weight,
  inImage,   
  outData);
  
  cudaMemcpy(this->input.data(),outData,nSize,cudaMemcpyDeviceToHost);
   for(int i=0;i<outSizeX;i++) this->out.data[i]=activator_function(this->input[i]);
  
}


  
__global__
void calCuda(
  float* inData,
  float* nextGrad,
  float* gradArray,
  float* weights,
  float* grads,
  int inSizeX,
  int inSizeY,
  int inSizeZ,
  int outSizeX
  ){
  int cuda_x = blockIdx.x * blockDim.x + threadIdx.x; 
  int cuda_y = blockIdx.y * blockDim.y + threadIdx.y; 
  int step_x = blockDim.x * gridDim.x; 
  int step_y = blockDim.y * gridDim.y; 
  
  for (int n = 0; n < outSizeX; n++) {
      gradArray[n] = nextGrad[n] * activator_derivativeCUDA(inData[n]);
      //printf("---%f\n",gradArray[0]);
      for (int i = cuda_x; i < inSizeX; i+=step_x)
        for (int j = cuda_y; j < inSizeY; j+=step_y)
          for (int z = 0; z < inSizeZ; z++) {
            int m = z*inSizeY*inSizeX+j*inSizeX+i;
            float w = weights[n*(inSizeX*inSizeY*inSizeZ)+m];
            
            //printf("-%f\n",nextGrad[n]);
            //printf("--%f\n",activator_derivativeCUDA(inData[n]));
            //printf("---%f\n",gradArray[n]);
            //printf("-----%f\n",w);
            grads[z*inSizeY*inSizeX+j*inSizeX+i] += nextGrad[n] * activator_derivativeCUDA(inData[n]) * w;
            
            //printf("-----------%f\n",grads[z*inSizeY*inSizeX+j*inSizeX+i]);
          }
    }
}

void fc_layer_t::calc_grads(tensor_t<float> &grad_next_layer){
    grads_in.clear();
    
    float* nextGrad = NULL;
    float gradArray[gradients.size()];
    int nextSize = sizeof(float)*grad_next_layer.data.size();
    
    memset(gradArray,0,sizeof(gradArray));
    cudaMalloc(&grads,this->inputSize);
    cudaMalloc(&nextGrad,nextSize);
    
    cudaMemcpy(nextGrad,grad_next_layer.data.data(),nextSize,cudaMemcpyHostToDevice);
    cudaMemcpy(inData,input.data(),sizeof(float)*input.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(Weight,weights.data.data(),weightSize,cudaMemcpyHostToDevice);
    
    int blockSize = 32;
    dim3 block(blockSize,blockSize);
    dim3 grid((this->in.size.x+blockSize-1)/blockSize,(this->in.size.y+blockSize-1)/blockSize);
    
    calCuda<<<grid,block>>>(
    inData,
    nextGrad,
    temp,
    Weight,
    grads,
    inSizeX,
    inSizeY,
    inSizeZ,
    outSizeX
    );
    
    cudaMemcpy(this->grads_in.data.data(),grads,inputSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(gradArray,temp,sizeof(float)*gradients.size(),cudaMemcpyDeviceToHost);
    
    for(int i=0;i<outSizeX;i++){
      gradients[i].grad = gradArray[i];
      }
    
}
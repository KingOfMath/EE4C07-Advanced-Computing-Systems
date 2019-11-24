#pragma once

#include <cstdint>
#include <cassert>
#include <cmath>

#include "gradient_t.h"
#include "layer_t.h"
#include "optimization_method.h"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include "device_launch_parameters.h"

#pragma pack(push, 1) 

struct conv_layer_t : layer_t { 
  std::vector<tensor_t<float>> filters;
  std::vector<tensor_t<gradient_t>> filter_grads;
  uint16_t stride;
  uint16_t extend_filter;
  
  int fSize;  //0
  int inSizeZ = this->in.size.z;  //1
  int outSizeX = this->out.size.x; //24
  int outSizeY = this->out.size.y; //24
  
  float* input = NULL;
  float* output = NULL;
  float* filterData = NULL;
  float* oldGrads = NULL;
  float* grads = NULL;
  
    float* nextGrads = NULL;
  float* gg = NULL;
  
  int gX,gY,gZ;
  
  int inputSize = sizeof(float)*this->in.data.size(); //3136
  int outputSize = sizeof(float)*outSizeX*outSizeY; //2304
  int filterSize; //100
  int fullFilterSize; //800
  int oldGradSize;
  int gradSize;
  
  

  conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size);

  // calculate the input position during convolution while know output position
  point_t map_to_input(point_t out, int z) {
    out.x *= stride;
    out.y *= stride;
    out.z = z;
    return out;
  }
  
  struct range_t {
    int min_x, min_y, min_z;
    int max_x, max_y, max_z;
  };

  int normalize_range(float f, int max, bool lim_min) {
    if (f <= 0)
      return 0;
    max -= 1;
    if (f >= max)
      return max;

    if (lim_min) // left side of inequality
      return static_cast<int>(std::ceil(f));
    else
      return static_cast<int>(std::floor(f));
  }

  // calculate the output position during convolution while know input position
  range_t map_to_output(int x, int y) {
    float a = x;
    float b = y;
    return
        {
            normalize_range((a - extend_filter + 1) / stride, out.size.x, true),
            normalize_range((b - extend_filter + 1) / stride, out.size.y, true),
            0,
            normalize_range(a / stride, out.size.x, false),
            normalize_range(b / stride, out.size.y, false),
            (int) filters.size() - 1,
        };
  }
  
  void activate(tensor_t<float> &in);

  // this is for the weight update, i.e, the filters
  void fix_weights();

  // this is for the backward path of convolution
  void calc_grads(tensor_t<float> &grad_next_layer);/*
   {

    float zero[8*25];
    memset(zero,0,sizeof(zero));

    for (int x = 0; x < in.size.x; x++) {
      for (int y = 0; y < in.size.y; y++) {
        range_t rn = map_to_output(x, y);
        for (int z = 0; z < in.size.z; z++) {
          //float sum_error = 0;
          for (int i = rn.min_x; i <= rn.max_x; i++) {
            int minx = i * stride;
            for (int j = rn.min_y; j <= rn.max_y; j++) {
              int miny = j * stride;
              for (int k = rn.min_z; k <= rn.max_z; k++) {
                //int w_applied = static_cast<int>(filters[k].get(x - minx, y - miny, z));
                //sum_error += w_applied * grad_next_layer(i, j, k);
                
                
                float v1 = in.data[z*(in.size.x*in.size.y)+y*(in.size.x)+x];
                float v2 = grad_next_layer.data[k*24*24+j*24+i];
                float v = v1*v2;
                
                //filter_grads[k].get(x - minx, y - miny, z).grad += v;
                
                zero[(k)*(extend_filter*extend_filter)+(y-miny)*(extend_filter)+(x-minx)] += v;
                
              }
            }
          }
          //grads_in(x, y, z) = sum_error;
        }
      }
    }
    
  for (int k = 0; k < filter_grads.size(); k++) {
      for (int i = 0; i < extend_filter; i++)
        for (int j = 0; j < extend_filter; j++)
          for (int z = 0; z < in.size.z; z++)
            filter_grads[k].get(i, j, z).grad = zero[k*(extend_filter*extend_filter)+(j)*(extend_filter)+(i)];
    
  }
  }
  
  */

  
 

};


#pragma pack(pop)
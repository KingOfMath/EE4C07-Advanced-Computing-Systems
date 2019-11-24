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

#include <iostream>
#include "../baseline/water.hpp"
#include "water_cuda.hpp"
#include "../utils/Timer.hpp"
#include "imgproc_cuda.hpp"


__host__ 
void initCUDAKernel(){
  int* i;
  cudaMalloc(&i,1);
}

std::shared_ptr<Histogram> runHistogramStageCUDA(const Image *previous, const WaterEffectOptions *options){

  auto hist = std::make_shared<Histogram>(getHistogramCUDA(previous));

  if (options->save_intermediate) {
    auto hist_img = hist->toImage();
    hist_img->toPNG("output/" + options->img_name + "_CUDAhistogram.png");
  }
  
  return hist;

}

std::shared_ptr<Image> runEnhanceStageCUDA(const Image *previous, const Histogram *hist, const WaterEffectOptions *options){
  if ((!options->histogram) || (hist == nullptr)) {
    throw std::runtime_error("Contrast enhancement is only possible when histogram stage has been performed.");
  }

  auto threshold = (int) (hist->max(0) * 0.1);
  auto img_enhanced = std::make_shared<Image>(previous->width, previous->height);
  enhanceContrastLinearlyCUDA(previous, hist, img_enhanced.get(), threshold, threshold);
  if (options->save_intermediate)
    img_enhanced->toPNG("output/" + options->img_name + "_enhanced.png");
  previous = img_enhanced.get();
  if (options->enhance_hist) {
    auto enhanced_hist = getHistogramC(img_enhanced.get());
    auto enhanced_hist_img = enhanced_hist.toImage();
    enhanced_hist_img->toPNG("output/" + options->img_name + "_enhanced_histogram.png");
  }

  return img_enhanced;                                    
}

std::shared_ptr<Image> runRippleStageCUDA(const Image *previous, const WaterEffectOptions *options){
  auto img_rippled = std::make_shared<Image>(previous->width, previous->height);
  applyRippleCUDA(previous, img_rippled.get(), options->ripple_frequency);
  if (options->save_intermediate)
    img_rippled->toPNG("output/" + options->img_name + "_CUDArippled.png");
  return img_rippled;
}

std::shared_ptr<Image> runBlurStageCUDA(const Image *previous, const WaterEffectOptions *options){
  
  Kernel gaussian = Kernel::gaussian(options->blur_size, options->blur_size, 1.0);
  auto img_blurred = std::make_shared<Image>(previous->width, previous->height);

  convoluteCUDA(previous, img_blurred.get(), &gaussian);

  if (options->save_intermediate)
    img_blurred->toPNG("output/" + options->img_name + "_CUDAblurred.png");

  return img_blurred;
}

std::shared_ptr<Image> runWaterEffectCUDA(const Image *src, const WaterEffectOptions *options) {

  Timer ts;
  std::shared_ptr<Histogram> hist;
  std::shared_ptr<Image> img_result;
  
  initCUDAKernel();

  if (options->histogram) {
    ts.start();
    hist = runHistogramStageCUDA(src, options);
    ts.stop();
    double t1 = ts.seconds();
    
    std::cout << "Stage: Histogram:        " << t1 << " s." << std::endl;
    //std::cout << "Stage: KernelTime:        " << KernelTime[0] << std::endl;
  }

  if (options->enhance) {
    ts.start();
    if (hist == nullptr) {
      throw std::runtime_error("Cannot run enhance stage without histogram.");
    }
    img_result = runEnhanceStageCUDA(src, hist.get(), options);
    ts.stop();
    std::cout << "Stage: Contrast enhance: " << ts.seconds() << " s." << std::endl;
    //std::cout << "Stage: KernelTime:        " << KernelTime[1] << "s." << std::endl;
  }

  if (options->ripple) {
  
    ts.start();
    if (img_result == nullptr) {
      img_result = runRippleStageCUDA(src, options);
    } else {
      img_result = runRippleStageCUDA(img_result.get(), options);
    }
    ts.stop();
    std::cout << "Stage: Ripple effect:    " << ts.seconds() << " s." << std::endl;
    //std::cout << "Stage: KernelTime:             " << KernelTime[2] << " s." << std::endl;
  }

  if (options->blur) {

    ts.start();
    if (img_result == nullptr) {
      img_result = runBlurStageCUDA(src, options);
    } else {
      img_result = runBlurStageCUDA(img_result.get(), options);
    }
    ts.stop();
    double t1 = ts.seconds();
    std::cout << "Stage: Blur:             " << t1 << " s." << std::endl;
    //std::cout << "Stage: KernelTime:             " << KernelTime[3] << " s." << std::endl;
  }
  
  /* */  
 
  
  return img_result;
}
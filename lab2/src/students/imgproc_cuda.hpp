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

#pragma once

#include "../utils/Image.hpp"
#include "../utils/Kernel.hpp"
#include "../utils/Histogram.hpp"
#include "../utils/Timer.hpp"

void convoluteCUDA(const Image *src, Image *dest, const Kernel *kernel);
Histogram getHistogramCUDA(const Image *src);
void enhanceContrastLinearlyCUDA(const Image *src, const Histogram *src_hist, Image *dest, int low, int high);
void applyRippleCUDA(const Image *src, Image *dest, float intensity);
void copyChannelCUDA(const Image *src, Image *dest, int channel);

Histogram getHistogramC(const Image *src);

extern double KernelTime[4];
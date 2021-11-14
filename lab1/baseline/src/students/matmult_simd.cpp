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

#include "../matmult.hpp"
#include "../utils/Timer.hpp"

// Intel intrinsics for SSE/AVX:
#include <immintrin.h>
#include <time.h>
/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O1")
/*************************************/

#define TIME_STEP_4 4
#define TIME_STEP_8 8

typedef union _avxd {
  __m256d val;
  __m128 val_128;
  double arr[4];
  float arr_128[4];
} avxd;

Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {

  size_t N,I,M;
  N = a.rows;
  M = b.columns;
  I = a.columns;

  if(I!=b.rows) 
	throw std::domain_error("Matrix dimensions do not match.");

  auto res_2 = Matrix<float>(N,M);
  __m128 x,y,z;
  avxd av;

if(N>=256){ 
  float** xa = new float*[1024];
  for(int i=0;i<1024;i++)
	xa[i] = new float[1024];
  float** xb = new float*[1024];
  for(int i=0;i<1024;i++)
	xb[i] = new float[1024];
  for(size_t i=0;i<N;i++)
	for(size_t j=0;j<I;j++)
		 xa[int(i)][int(j)] = a(i,j);
  for(size_t i=0;i<I;i++)
	for(size_t j=0;j<M;j++)
		 xb[int(j)][int(i)] = b(i,j);
 for(size_t i=0;i<N;i++){
	for(size_t j=0;j<M;j++){
		res_2(i,j) = 0;
		z = _mm_setzero_ps();
		for(size_t k=0;k<I-TIME_STEP_4;k+=TIME_STEP_4){
			x = _mm_loadu_ps(xa[int(i)]+k);
			y = _mm_loadu_ps(xb[int(j)]+k);
			x = _mm_mul_ps(x,y);
			z = _mm_add_ps(x,z);
		}
		av.val_128 = z;
		for(size_t m=0;m<4;m++)
			res_2(i,j) += av.arr_128[m];  
		for(size_t k=I-(I%TIME_STEP_4);k<I;k++)
			res_2(i,j) += xa[int(i)][int(k)]*xb[int(j)][int(k)];
	}
  }
} else {
 for(size_t i=0;i<N;i++){
	for(size_t j=0;j<M;j++){
		res_2(i,j) = 0;
		z = _mm_setzero_ps();
		for(size_t k=0;k<I-TIME_STEP_4;k+=TIME_STEP_4){
			x = _mm_set_ps(a(i,k),a(i,k+1),a(i,k+2),a(i,k+3));
			y = _mm_set_ps(b(k,j),b(k+1,j),b(k+2,j),b(k+3,j));
			x = _mm_mul_ps(x,y);
			z = _mm_add_ps(x,z);
		}
		av.val_128 = z;
		for(size_t m=0;m<4;m++)
			res_2(i,j) += av.arr_128[m];  
		for(size_t k=I-(I%TIME_STEP_4);k<I;k++)
			res_2(i,j) += a(i,k)*b(k,j);
  	}
  }
}
  return res_2;
}

Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  size_t N,I,M;
  N = a.rows;
  M = b.columns;
  I = a.columns;

  if(I!=b.rows) 
	throw std::domain_error("Matrix dimensions do not match.");

  auto res_2 = Matrix<double>(N*4,M*4);
  avxd x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15;


if(N>=256){
  double** xa = new double*[1024];
  for(int i=0;i<1024;i++)
	xa[i] = new double[1024];
  double** xb = new double*[1024];
  for(int i=0;i<1024;i++)
	xb[i] = new double[1024];
  for(size_t i=0;i<N;i++)
	for(size_t j=0;j<I;j++)
		 xa[int(i)][int(j)] = a(i,j);
  for(size_t i=0;i<I;i++)
	for(size_t j=0;j<M;j++)
		 xb[int(j)][int(i)] = b(i,j);

  for(size_t j=0;j<M;j+=TIME_STEP_4){
        for(size_t i=0;i<N;i+=TIME_STEP_4){
       		z0.val=z1.val=z2.val=z3.val=z4.val=z5.val=z6.val=z7.val=z8.val=z9.val=z10.val=z11.val=z12.val=z13.val=z14.val=z15.val=_mm256_setzero_pd();
        	for(size_t k=0;k<I-TIME_STEP_4;k+=TIME_STEP_4){
        		x0.val = _mm256_loadu_pd(xa[int(i)]+k);
        		x1.val = _mm256_loadu_pd(xa[int(i+1)]+k);
        		x2.val = _mm256_loadu_pd(xa[int(i+2)]+k);
        		x3.val = _mm256_loadu_pd(xa[int(i+3)]+k);
        		y0.val = _mm256_loadu_pd(xa[int(j)]+k);
        		y1.val = _mm256_loadu_pd(xa[int(j+1)]+k);
        		y2.val = _mm256_loadu_pd(xa[int(j+2)]+k);
        		y3.val = _mm256_loadu_pd(xa[int(j+3)]+k);
        		z0.val = _mm256_add_pd(z0.val,_mm256_mul_pd(x0.val,y0.val));
        		z1.val = _mm256_add_pd(z1.val,_mm256_mul_pd(x0.val,y1.val));
        		z2.val = _mm256_add_pd(z2.val,_mm256_mul_pd(x0.val,y2.val));
        		z3.val = _mm256_add_pd(z3.val,_mm256_mul_pd(x0.val,y3.val));
        		z4.val = _mm256_add_pd(z4.val,_mm256_mul_pd(x1.val,y0.val));
        		z5.val = _mm256_add_pd(z5.val,_mm256_mul_pd(x1.val,y1.val));
        		z6.val = _mm256_add_pd(z6.val,_mm256_mul_pd(x1.val,y2.val));
        		z7.val = _mm256_add_pd(z7.val,_mm256_mul_pd(x1.val,y3.val));
        		z8.val = _mm256_add_pd(z8.val,_mm256_mul_pd(x2.val,y0.val));
        		z9.val = _mm256_add_pd(z9.val,_mm256_mul_pd(x2.val,y1.val));
        		z10.val = _mm256_add_pd(z10.val,_mm256_mul_pd(x2.val,y2.val));
        		z11.val = _mm256_add_pd(z11.val,_mm256_mul_pd(x2.val,y3.val));
        		z12.val = _mm256_add_pd(z12.val,_mm256_mul_pd(x3.val,y0.val));
        		z13.val = _mm256_add_pd(z13.val,_mm256_mul_pd(x3.val,y1.val));
        		z14.val = _mm256_add_pd(z14.val,_mm256_mul_pd(x3.val,y2.val));
        		z15.val = _mm256_add_pd(z15.val,_mm256_mul_pd(x3.val,y3.val));
      	}  	
		for(size_t m=0;m<4;m++){
			res_2(i,j) += z0.arr[m];
			res_2(i,j+1) += z1.arr[m];
			res_2(i,j+2) += z2.arr[m];
			res_2(i,j+3) += z3.arr[m];
			res_2(i+1,j) += z4.arr[m];
			res_2(i+1,j+1) += z5.arr[m];
			res_2(i+1,j+2) += z6.arr[m];
			res_2(i+1,j+3) += z7.arr[m];
			res_2(i+2,j) += z8.arr[m];
			res_2(i+2,j+1) += z9.arr[m];
			res_2(i+2,j+2) += z10.arr[m];
			res_2(i+2,j+3) += z11.arr[m];
			res_2(i+3,j) += z12.arr[m];
			res_2(i+3,j+1) += z13.arr[m];
			res_2(i+3,j+2) += z14.arr[m];
			res_2(i+3,j+3) += z15.arr[m];
		}	
      }
  }
} else {

  for(size_t j=0;j<M;j+=TIME_STEP_4){
        for(size_t i=0;i<N;i+=TIME_STEP_4){
       		z0.val=z1.val=z2.val=z3.val=z4.val=z5.val=z6.val=z7.val=z8.val=z9.val=z10.val=z11.val=z12.val=z13.val=z14.val=z15.val=_mm256_setzero_pd();
        	for(size_t k=0;k<I-TIME_STEP_4;k+=TIME_STEP_4){
       			x0.val = _mm256_set_pd(a(i,k),a(i,k+1),a(i,k+2),a(i,k+3));
       			x1.val = _mm256_set_pd(a(i+1,k),a(i+1,k+1),a(i+1,k+2),a(i+1,k+3));
       			x2.val = _mm256_set_pd(a(i+2,k),a(i+2,k+1),a(i+2,k+2),a(i+2,k+3));
  	     		x3.val = _mm256_set_pd(a(i+3,k),a(i+3,k+1),a(i+3,k+2),a(i+3,k+3));
        	     	y0.val = _mm256_set_pd(b(k,j),b(k+1,j),b(k+2,j),b(k+3,j));
       			y1.val = _mm256_set_pd(b(k,j+1),b(k+1,j+1),b(k+2,j+1),b(k+3,j+1));
       			y2.val = _mm256_set_pd(b(k,j+2),b(k+1,j+2),b(k+2,j+2),b(k+3,j+2));
       			y3.val = _mm256_set_pd(b(k,j+3),b(k+1,j+3),b(k+2,j+3),b(k+3,j+3));
        		z0.val = _mm256_add_pd(z0.val,_mm256_mul_pd(x0.val,y0.val));
        		z1.val = _mm256_add_pd(z1.val,_mm256_mul_pd(x0.val,y1.val));
        		z2.val = _mm256_add_pd(z2.val,_mm256_mul_pd(x0.val,y2.val));
        		z3.val = _mm256_add_pd(z3.val,_mm256_mul_pd(x0.val,y3.val));
        		z4.val = _mm256_add_pd(z4.val,_mm256_mul_pd(x1.val,y0.val));
        		z5.val = _mm256_add_pd(z5.val,_mm256_mul_pd(x1.val,y1.val));
        		z6.val = _mm256_add_pd(z6.val,_mm256_mul_pd(x1.val,y2.val));
        		z7.val = _mm256_add_pd(z7.val,_mm256_mul_pd(x1.val,y3.val));
        		z8.val = _mm256_add_pd(z8.val,_mm256_mul_pd(x2.val,y0.val));
        		z9.val = _mm256_add_pd(z9.val,_mm256_mul_pd(x2.val,y1.val));
        		z10.val = _mm256_add_pd(z10.val,_mm256_mul_pd(x2.val,y2.val));
        		z11.val = _mm256_add_pd(z11.val,_mm256_mul_pd(x2.val,y3.val));
        		z12.val = _mm256_add_pd(z12.val,_mm256_mul_pd(x3.val,y0.val));
        		z13.val = _mm256_add_pd(z13.val,_mm256_mul_pd(x3.val,y1.val));
        		z14.val = _mm256_add_pd(z14.val,_mm256_mul_pd(x3.val,y2.val));
        		z15.val = _mm256_add_pd(z15.val,_mm256_mul_pd(x3.val,y3.val));
      	}  	
		for(size_t m=0;m<4;m++){
			res_2(i,j) += z0.arr[m];
			res_2(i,j+1) += z1.arr[m];
			res_2(i,j+2) += z2.arr[m];
			res_2(i,j+3) += z3.arr[m];
			res_2(i+1,j) += z4.arr[m];
			res_2(i+1,j+1) += z5.arr[m];
			res_2(i+1,j+2) += z6.arr[m];
			res_2(i+1,j+3) += z7.arr[m];
			res_2(i+2,j) += z8.arr[m];
			res_2(i+2,j+1) += z9.arr[m];
			res_2(i+2,j+2) += z10.arr[m];
			res_2(i+2,j+3) += z11.arr[m];
			res_2(i+3,j) += z12.arr[m];
			res_2(i+3,j+1) += z13.arr[m];
			res_2(i+3,j+2) += z14.arr[m];
			res_2(i+3,j+3) += z15.arr[m];
		}
	}
  }

}

  return res_2;

}

/*************************************/
#pragma GCC pop_options
/*************************************/

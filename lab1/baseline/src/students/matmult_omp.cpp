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

// OpenMP:
#include <omp.h>

#include "../matmult.hpp"

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O1")
/*************************************/

Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num) {
  auto result = Matrix<float>(a.rows,b.columns);
  size_t N = a.rows;
  size_t I = a.columns;
  size_t M = b.columns;

  size_t i,j,k;
if(N>64){
  float * xa = new float[N*I];
  float * xb = new float[I*M];

#pragma omp parallel for
  for(i=0;i<a.rows;i++)
	for(j=0;j<a.columns;j++){
		xa[i*N+j] = a(i,j);	
		xb[j*M+i] = b(i,j);
	}
  float t1,t2,t3,t4;  
#pragma omp parallel shared(result) private(i,j,k,t1,t2,t3,t4) num_threads(num)
{
  for (i=0;i<N;i++){
	for(j=0;j<M;j++){
		for(k=0;k<I-4;k+=4){
			t1 = xa[i*N+k] * xb[j*M+k]; 
			t2 = xa[i*N+k+1] * xb[j*M+k+1]; 
			t3 = xa[i*N+k+2] * xb[j*M+k+2]; 
			t4 = xa[i*N+k+3] * xb[j*M+k+3]; 		
			result(i,j) += t1+t2+t3+t4;
			}
		}
		for(k=I-(I%4);k<I;k++)
			result(i,j) += a(i,k) * b(k,j);
  }
}
} else {

#pragma omp parallel shared(result) private(i,j,k) num_threads(num)
{
  for (i=0;i<N;i++){
	for(j=0;j<M;j++){
		for(k=0;k<I;k++){
			result(i,j) += a(i,k) * b(k,j);
		}
//		for(k=0;k<I-4;k+=4){
//			t1 = a(i,k) * b(k,j); 
//			t2 = a(i,k+1) * b(k+1,j); 
//			t3 = a(i,k+2) * b(k+2,j); 
//			t4 = a(i,k+3) * b(k+3,j); 		
//			result(i,j) += t1+t2+t3+t4;
//			}
//		for(k=I-(I%4);k<I;k++)
//			result(i,j) += a(i,k) * b(k,j);
  }
}

}
}
  return result;
}


Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num) {
  auto result = Matrix<double>(a.rows,b.columns);
  size_t N = a.rows;
  size_t I = a.columns;
  size_t M = b.columns;

  size_t i,j,k;
  double * xa = new double[N*I];
  double * xb = new double[I*M];

#pragma omp parallel for
  for(i=0;i<a.rows;i++)
	for(j=0;j<a.columns;j++){
		xa[i*N+j] = a(i,j);	
		xb[j*M+i] = b(i,j);
	}
  double t1,t2,t3,t4;  
#pragma omp parallel shared(result) private(i,j,k,t1,t2,t3,t4) num_threads(num)
{
  for (i=0;i<N;i++){
	for(j=0;j<M;j++){
		for(k=0;k<I-4;k+=4){
			t1 = xa[i*N+k] * xb[j*M+k]; 
			t2 = xa[i*N+k+1] * xb[j*M+k+1]; 
			t3 = xa[i*N+k+2] * xb[j*M+k+2]; 
			t4 = xa[i*N+k+3] * xb[j*M+k+3]; 		
			result(i,j) += t1+t2+t3+t4;
			}
		}
		for(k=I-(I%4);k<I;k++)
			result(i,j) += a(i,k) * b(k,j);
  }
}
  for (size_t i=0;i<a.rows;i++){
	for(size_t j=0;j<b.columns;j++){
		for(size_t k=0;k<a.columns;k++){
			result(i,j) += a(i,k) * b(k,j);
		}
	}
  }
  return result;
}
#pragma GCC pop_options

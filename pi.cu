
#include <iostream>
using namespace std;
#include <iomanip>
#include <cassert>
#include <helper_cuda.h>
#include "type.h"
#include "fort.h"

__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

inline bool is_pow_2(int x) { return ((x&(x-1)) == 0); }

__constant__ __device__ Real step;

__host__ __device__ Real myf(const Real &x)
{
  return _one_/(_one_ + x*x);
}

template<int block_size> __device__ void _calculate_pi_with_device_(Real *block_sum, int n_grids)
{
  extern __shared__ Real s_data [];
  
  int n = n_grids/gridDim.x;
  
  const Real x_left = blockIdx.x*n*step;
  
  // number of grid points of the last block 
  if(blockIdx.x == gridDim.x-1)
    n = n_grids - n*(gridDim.x-1);
  
  Real s = _zero_;
  for(int i = 0; i < n; i += blockDim.x) {
    const int j = i + threadIdx.x;
    if(j < n) {
      const Real x = x_left + j*step;
      s += myf(x);
    }
  }
  s_data[threadIdx.x] = s;
  __syncthreads();
  
  // do reduction in shared memory
  for(int i = blockDim.x/2; i > 0; i /= 2) {
    if(threadIdx.x < i)
      s_data[threadIdx.x] += s_data[threadIdx.x + i];
    __syncthreads();
  }
  
  if(threadIdx.x == 0) {
    atomicAdd(block_sum, s_data[0]);
    //printf("Block %d, sum = %f\n", blockIdx.x, s_data[0]*step);
  }
}

__global__ void calculate_pi_with_device_wrapper(const int block_dim, Real *block_sum, int n_grids)
{

  switch (block_dim) {

  case 2048:
    _calculate_pi_with_device_<2048>(block_sum, n_grids);
    break;
    
  case 1024:
    _calculate_pi_with_device_<1024>(block_sum, n_grids);
    break;
    
  case 512:
    _calculate_pi_with_device_<512>(block_sum, n_grids);
    break;

  case 256:
    _calculate_pi_with_device_<256>(block_sum, n_grids);
    break;

  case 128:
    _calculate_pi_with_device_<128>(block_sum, n_grids);
    break;

  case 64:
    _calculate_pi_with_device_<64>(block_sum, n_grids);
    break;

  case 32:
    _calculate_pi_with_device_<32>(block_sum, n_grids);
    break;

  case 16:
    _calculate_pi_with_device_<16>(block_sum, n_grids);
    break;
    
  case 8:
    _calculate_pi_with_device_<8>(block_sum, n_grids);
    break;

  case 4:
    _calculate_pi_with_device_<4>(block_sum, n_grids);
    break;

  case 2:
    _calculate_pi_with_device_<2>(block_sum, n_grids);
    break;    

  case 1:
    _calculate_pi_with_device_<1>(block_sum, n_grids);
    break;

  default:
    break;
  }
}

void calculate_pi_with_device(const int block_dim, const int grid_dim)
{
  assert(is_pow_2(block_dim));

  cout << " sizeof(Real) = " << sizeof(Real) << endl;
  
  const int n = _NGrids_;
  
  cout << " n = " << n << ", block_dim = " << block_dim 
       << ", grid_dim = " << grid_dim << endl;

  const Real dx = _one_/(n-1);

  checkCudaErrors(cudaMemcpyToSymbol(step, &dx, sizeof(Real)));

  Real *block_sum = 0;
  checkCudaErrors(cudaMalloc((void **) &block_sum, sizeof(Real)));
  checkCudaErrors(cudaMemset(block_sum, 0, sizeof(Real)));


  const int share_memory_size = sizeof(Real)*block_dim;
  calculate_pi_with_device_wrapper<<<grid_dim, block_dim, share_memory_size>>>(block_dim, block_sum, n);

  Real block_sum_host = 0.0;
  checkCudaErrors(cudaMemcpy(&block_sum_host, block_sum, sizeof(Real), cudaMemcpyDeviceToHost));
  
  if(block_sum) { checkCudaErrors(cudaFree(block_sum)); block_sum = 0; }
  
  Real pi = block_sum_host;
  pi -= _pt5_ * (myf(_zero_) + myf(_one_));
  pi *= 4*dx;

  cout << "        Pi = " << setprecision(16) << pi << "\n"
       << " analytical: " << setprecision(16) << 2.0*asin(1.0) << endl;
}

// Fotran version: CalculatePiWithDevice
extern "C" void FORT(calculatepiwithdevice)(const int &block_dim, const int &grid_dim)
{
  calculate_pi_with_device(block_dim, grid_dim);
}


#include <iostream>
using namespace std;

#include <cassert>

#include <driver_types.h>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
//#include <cutil.h>

#define CUDA_SAFE_CALL(x) checkCudaErrors(x)

void print_GPU_information()
{
  cout << "\n"
       << " ************************************\n"
       << " **     GPU devices information    **\n"
       << " ************************************\n"
       << endl;
  
  int n_dev = -1;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&n_dev));
  cout << " There "
       << (n_dev == 1 ? "is " : "are ")
       << n_dev << " GPU device";
  if(n_dev > 1) cout << "s";
  cout << "\n" << endl;
  
  cudaDeviceProp *dp = new cudaDeviceProp;
  assert(dp);

  for(int idev = 0; idev < n_dev; idev++) {
    CUDA_SAFE_CALL(cudaGetDeviceProperties(dp, idev));

    cout << "\n"
	 << " ** GPU index: " << idev << "\n"
	 << " name: " << dp->name << "\n"
	 << " totalGlobalMem: " << dp->totalGlobalMem << "\n"
	 << " sharedMemPerBlock: " << dp->sharedMemPerBlock << "\n"
	 << " regsPerBlock: " << dp->regsPerBlock << "\n"
	 << " warpSize: " << dp->warpSize << "\n"
	 << " memPitch: " << dp->memPitch << "\n"
	 << " maxThreadsPerBlock: " << dp->maxThreadsPerBlock << "\n"
	 << " maxThreadsDim: " << dp->maxThreadsDim[0] 
	 << " " << dp->maxThreadsDim[1] << "  " << dp->maxThreadsDim[2] << "\n"
	 << " maxGridSize: " << dp->maxGridSize[0] 
	 << " " << dp->maxGridSize[1] << " " << dp->maxGridSize[2] << "\n"
	 << " totalConstMem: " << dp->totalConstMem << "\n"
	 << " major: " << dp->major << "\n"
	 << " minor: " << dp->minor << "\n"
	 << " clockRate: " << dp->clockRate << "\n"
	 << " textureAlignment: " << dp->textureAlignment << "\n"
	 << " deviceOverlap: " << dp->deviceOverlap << "\n"
	 << " multiProcessorCount: " << dp->multiProcessorCount 
	 << endl;
  }
  
  if(dp) { delete dp; dp = 0; }
}

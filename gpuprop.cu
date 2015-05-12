
#include <iostream>
using namespace std;
#include <helper_cuda.h>

void gpu_test()
{
  cout << endl;

  int deviceCount = -1;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  cout << " deviceCount " << deviceCount  << endl;
  
  int dev = 0;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
  cout << " Using device: " << dev << "\n"
       << " Name: " << prop.name << ","
       << " Global mem: " << prop.totalGlobalMem/1024.0/1024/1024 << ","
       << " Compute v" << (int) prop.major << "." << (int)prop.minor << ","
       << " Clock: " << (int) prop.clockRate << " KHz" << endl;
  
  cout << endl;
}

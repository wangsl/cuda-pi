
#include <cstdlib>
#include <iostream>
using namespace std;
#include "timer.h"
#include "fort.h"

void calculate_pi_with_device(const int block_dim, const int grid_dim);

void pi_with_openmp();

void gpu_test();

extern "C" void FORT(fortrantest)(const int &block_dim, const int &grid_dim);

int main(int argc, const char * argv[])
{
  gpu_test();

  int block_dim = 512;
  int grid_dim = 512;
  int n = 1;
  int op = 0;

  if(argc > 1) 
    block_dim = atoi(argv[1]);
  
  if(argc > 2)
    grid_dim = atoi(argv[2]);

  if(argc > 3)
    n = atoi(argv[3]);

  if(argc > 4)
    op = atoi(argv[4]);
  
  Timer timer;

  Timer::print_time(timer.time());
  
  for(int i = 0; i < n; i++) {
    cout << "\n *** " << i << " ***" << endl;
    
    if(op == 0 || op == 1) {
      timer.reset();
      FORT(fortrantest)(block_dim, grid_dim);
      cout << " Wall time for GPU: ";
      Timer::print_time(timer.time());
    }
    
    if(op == 0 || op == 2) {
      timer.reset();
      pi_with_openmp();
      cout << " Wall time for OpenMP: ";
      Timer::print_time(timer.time());
    }
  }

  return 0;
}

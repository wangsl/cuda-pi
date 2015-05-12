
#include <iostream>
using namespace std;
#include <iomanip>
#include <cmath>
#include <omp.h>

#include "type.h"

void pi_with_openmp()
{
  cout << " Pi with OpenMP" << endl;

  const int num_steps = _NGrids_;

  cout << " n = " << num_steps << endl;

  const double step = 1.0/double(num_steps - 1);
  
  double sum = 0.0;
#pragma omp parallel for if(num_steps > 100)	\
  default(shared) schedule(dynamic, 100)	\
  reduction(+:sum)
  for(int i = 0; i < num_steps; i++) {
    double x = i*step;
    double x2 = x*x;
    sum += 1.0/(1.0+x2);
  }

  sum -= 0.5*(1.0 + 0.5);
  
  double pi = 4*step*sum;

  cout << "        Pi = " << setprecision(16) << pi << "\n"
       << " analytical: " << setprecision(16) << 2.0*asin(1.0) << endl;
}

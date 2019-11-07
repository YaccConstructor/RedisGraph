#include <thrust/detail/config/compiler_fence.h>
#include <cusp/coo_matrix.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/elementwise.h>

int main()
{

  cusp::coo_matrix<int, double, cusp::device_memory> B(2,2,4);

  B.values[0] = 1;
  B.row_indices[0] = 0;
  B.column_indices[0]= 0;

  B.values[1] = 2;
  B.row_indices[1] = 0;
  B.column_indices[1]= 1;

  B.values[2] = 3;
  B.row_indices[2] = 1;
  B.column_indices[2]= 0;

  B.values[3] = 4;
  B.row_indices[3] = 1;
  B.column_indices[3]= 1;


  cusp::coo_matrix<int,double, cusp::device_memory> D(B);
  cusp::coo_matrix<int,double, cusp::device_memory> C;

  thrust::identity<double> identity;
  thrust::multiplies<double> combine;
  thrust::plus<double>       reduce;


  cusp::multiply(B,B,C, identity, combine, reduce);
  cusp::add(C, D, D);

  cusp::print(B);
  cusp::print(C);
  cusp::print(D);

}
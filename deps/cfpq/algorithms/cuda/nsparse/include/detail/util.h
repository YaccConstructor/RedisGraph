#pragma once
#include <cmath>
#include <ostream>
#include <thrust/device_vector.h>

namespace nsparse {
namespace util {

template <typename T>
struct bucket_info_t {
  T row_id;
  T a_row_begin;
  T b_row_begin;
  T a_row_end;
  T b_row_end;
};

template <typename T>
inline T div(T m, T n) {
  return ceil((double)m / n);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T>& vec) {
  thrust::host_vector<T> h_vec = vec;
  for (auto item : h_vec) {
    os << item << " ";
  }
  return os;
}

}  // namespace util

}  // namespace nsparse
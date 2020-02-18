#pragma once

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace nsparse {

template <typename T>
__global__ void merge_count(thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
                            thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
                            thrust::device_ptr<T> rpt_c) {
  constexpr T invalid_value = std::numeric_limits<T>::max();

  auto row = blockIdx.x;

  T global_offset_a = rpt_a[row];
  T sz_a = rpt_a[row + 1] - global_offset_a;

  T global_offset_b = rpt_b[row];
  T sz_b = rpt_b[row + 1] - global_offset_b;

  __shared__ T buff_a[32];
  __shared__ T buff_b[32];

  T cur_a_pos = 0;
  T cur_b_pos = 0;

  T prev = invalid_value;
  T next = invalid_value;

  bool moved_a = true;
  bool moved_b = true;

  T counter = 0;

  for (auto iter = 0; iter < sz_a + sz_b; iter++) {
    if (moved_a && cur_a_pos % 32 == 0) {
      if (threadIdx.x + cur_a_pos < sz_a) {
        buff_a[threadIdx.x] = col_a[global_offset_a + threadIdx.x + cur_a_pos];
      }
      __syncthreads();
      moved_a = false;
    }
    if (moved_b && cur_b_pos % 32 == 0) {
      if (threadIdx.x + cur_b_pos < sz_b) {
        buff_b[threadIdx.x] = col_b[global_offset_b + threadIdx.x + cur_b_pos];
      }
      __syncthreads();
      moved_b = false;
    }

    if (cur_a_pos == sz_a) {
      next = buff_b[cur_b_pos % 32];
      cur_b_pos++;
      moved_b = true;
    } else if (cur_b_pos == sz_b) {
      next = buff_a[cur_a_pos % 32];
      cur_a_pos++;
      moved_a = true;
    } else {
      T nextA = buff_a[cur_a_pos % 32];
      T nextB = buff_b[cur_b_pos % 32];
      if (nextA < nextB) {
        next = nextA;
        cur_a_pos++;
        moved_a = true;
      } else {
        next = nextB;
        cur_b_pos++;
        moved_b = true;
      }
    }
    counter += next != prev;
    prev = next;
  }

  if (threadIdx.x == 0) {
    rpt_c[row] = counter;
  }
}

template <typename T>
__global__ void merge(thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
                      thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
                      thrust::device_ptr<const T> rpt_c, thrust::device_ptr<T> col_c) {
  constexpr T invalid_value = std::numeric_limits<T>::max();

  auto row = blockIdx.x;

  T global_offset_a = rpt_a[row];
  T sz_a = rpt_a[row + 1] - global_offset_a;

  T global_offset_b = rpt_b[row];
  T sz_b = rpt_b[row + 1] - global_offset_b;

  T global_offset_c = rpt_c[row];
  T sz_c = rpt_c[row + 1] - global_offset_c;

  __shared__ T buff_a[32];
  __shared__ T buff_b[32];
  __shared__ T buff_c[32];

  T cur_a_pos = 0;
  T cur_b_pos = 0;

  T prev = invalid_value;
  T next = invalid_value;

  bool moved_a = true;
  bool moved_b = true;

  T counter = 0;

  for (auto iter = 0; iter < sz_a + sz_b; iter++) {
    if (moved_a && cur_a_pos % 32 == 0) {
      if (threadIdx.x + cur_a_pos < sz_a) {
        buff_a[threadIdx.x] = col_a[global_offset_a + threadIdx.x + cur_a_pos];
      }
      __syncthreads();
      moved_a = false;
    }
    if (moved_b && cur_b_pos % 32 == 0) {
      if (threadIdx.x + cur_b_pos < sz_b) {
        buff_b[threadIdx.x] = col_b[global_offset_b + threadIdx.x + cur_b_pos];
      }
      __syncthreads();
      moved_b = false;
    }

    if (cur_a_pos == sz_a) {
      next = buff_b[cur_b_pos % 32];
      cur_b_pos++;
      moved_b = true;
    } else if (cur_b_pos == sz_b) {
      next = buff_a[cur_a_pos % 32];
      cur_a_pos++;
      moved_a = true;
    } else {
      T nextA = buff_a[cur_a_pos % 32];
      T nextB = buff_b[cur_b_pos % 32];
      if (nextA < nextB) {
        next = nextA;
        cur_a_pos++;
        moved_a = true;
      } else {
        next = nextB;
        cur_b_pos++;
        moved_b = true;
      }
    }

    if (next != prev) {
      if (threadIdx.x == 0) {
        buff_c[counter % 32] = next;
      }
      counter++;

      if (counter % 32 == 0) {
        // flush
        col_c[global_offset_c + counter - 32 + threadIdx.x] = buff_c[threadIdx.x];
      }
      __syncthreads();
    }
    prev = next;
  }

  if (threadIdx.x < (counter % 32)) {
    col_c[global_offset_c + counter - (counter % 32) + threadIdx.x] = buff_c[threadIdx.x];
  }
}

}  // namespace nsparse
#pragma once

#include <thrust/device_ptr.h>

#include <detail/bitonic.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace nsparse {

template <typename T>
__global__ void validate_order(thrust::device_ptr<const T> rpt_a,
                               thrust::device_ptr<const T> col_a) {
  auto row = blockIdx.x;

  T global_offset_a = rpt_a[row];
  T sz_a = rpt_a[row + 1] - global_offset_a;

  for (auto i = global_offset_a + threadIdx.x; i < sz_a + global_offset_a; i += blockDim.x) {
    if (i > global_offset_a)
      assert(col_a[i - 1] <= col_a[i]);

    assert(col_a[i] != std::numeric_limits<T>::max());
  }
}

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

template <typename T, T block_size>
__global__ void merge_path(thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
                           thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
                           thrust::device_ptr<T> rpt_c, thrust::device_ptr<T> col_c) {
  const auto row = blockIdx.x;

  constexpr T max_val = std::numeric_limits<T>::max();

  const T global_offset_a = rpt_a[row];
  const T sz_a = rpt_a[row + 1] - global_offset_a;

  const T global_offset_b = rpt_b[row];
  const T sz_b = rpt_b[row + 1] - global_offset_b;

  T global_offset_c = rpt_c[row];
  const T sz_c = rpt_c[row + 1] - global_offset_c;

  const T block_count = (sz_a + sz_b + block_size - 1) / block_size;

  T begin_a = 0;
  T begin_b = 0;

  __shared__ T raw_a[block_size + 2];
  __shared__ T raw_b[block_size + 2];
  __shared__ T res[block_size];

  bool dir = true;
  T item_from_prev_chank = max_val;


  for (auto i = 0; i < block_count; i++) {

    __shared__ T max_x_index;
    __shared__ T max_y_index;

    T max_x_index_per_thread = 0;
    T max_y_index_per_thread = 0;

    assert(sz_a >= begin_a);
    assert(sz_b >= begin_b);

    T buf_a_size = min(sz_a - begin_a, block_size);
    T buf_b_size = min(sz_b - begin_b, block_size);

    if (threadIdx.x == 0) {
      max_x_index = 0;
      max_y_index = 0;
    }

    __syncthreads();

    for (auto j = threadIdx.x; j < block_size + 2; j += blockDim.x) {
      if (j > 0 && j - 1 < buf_a_size) {
        raw_a[j] = col_a[global_offset_a + j - 1 + begin_a];
      } else {
        raw_a[j] = max_val;
      }
      if (j > 0 && j - 1 < buf_b_size) {
        raw_b[j] = col_b[global_offset_b + j - 1 + begin_b];
      } else {
        raw_b[j] = max_val;
      }
    }

    __syncthreads();

    const T to_process = min(buf_b_size + buf_a_size, block_size);

    for (auto j = threadIdx.x; j < to_process; j += blockDim.x) {
      const T ind_sum = j + 2;
      const T y = ind_sum;
      const T x = 0;
      const T iter_count = ind_sum;

      T l = 0;
      T r = iter_count;

      while (r - l > 1) {
        T step = (r - l) / 2;
        T check = l + step;

        assert(check != 0);
        assert(check != iter_count);

        T check_x = x + check;
        T check_y = y - check;

        bool ans = raw_b[check_y] > raw_a[check_x];

        l += step * ans;
        r -= step * !ans;
      }

      T ans_x = x + l;
      T ans_y = y - l;

      if (ans_y == 1 || ans_x == 0) {
        if (ans_y == 1) {
          res[j] = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        } else {
          res[j] = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        }
      } else {
        if (raw_b[ans_y - 1] > raw_a[ans_x]) {
          res[j] = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        } else {
          res[j] = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        }
      }
    }

    //    for (auto m = threadIdx.x; m < to_process; m += blockDim.x) {
    //      col_c[global_offset_c + m] = res[m];
    //    }

    atomicMax(&max_x_index, max_x_index_per_thread);
    atomicMax(&max_y_index, max_y_index_per_thread);

    __syncthreads();

    T counter = 0;

    if (dir) {
      for (auto m = threadIdx.x; m < to_process; m += blockDim.x) {
        if (m > 0)
          counter += (res[m] - res[m - 1]) != 0;
        else
          counter += (res[0] - item_from_prev_chank) != 0;
        item_from_prev_chank = res[m];
      }
    } else {
      for (auto m = blockDim.x - 1 - threadIdx.x; m < to_process; m += blockDim.x) {
        if (m > 0)
          counter += (res[m] - res[m - 1]) != 0;
        else
          counter += (res[0] - item_from_prev_chank) != 0;
        item_from_prev_chank = res[m];
      }
    }

    dir = !dir;

    atomicAdd(rpt_c.get() + row, counter);

    begin_a += max_x_index;
    begin_b += max_y_index;

    __syncthreads();

    global_offset_c += to_process;
  }
}

}  // namespace nsparse
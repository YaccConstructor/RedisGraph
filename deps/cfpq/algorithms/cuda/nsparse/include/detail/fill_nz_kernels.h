#pragma once

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename T>
__global__ void filter_hash_table(thrust::device_ptr<const T> row_index,
                                  thrust::device_ptr<const T> hash_table,
                                  thrust::device_ptr<const T> hash_table_offsets,
                                  thrust::device_ptr<const T> rows_in_table,
                                  thrust::device_ptr<T> col_index) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();
  auto i = blockIdx.x;
  T hash_table_size = hash_table_offsets[i + 1] - hash_table_offsets[i];
  T hash_table_offset = hash_table_offsets[i];

  T row_id = rows_in_table[i];

  T col_offset = row_index[row_id];

  __shared__ T offset;

  if (threadIdx.x == 0) {
    offset = 0;
  }

  __syncthreads();

  for (T j = threadIdx.x; j < hash_table_size; j += blockDim.x) {
    T value = hash_table[j + hash_table_offset];
    if (value != hash_invalidated) {
      T idx = atomicAdd(&offset, 1);
      col_index[col_offset + idx] = value;
    }
  }
}

template <typename T>
__global__ void fill_nz_block_row_global(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> rows_col,
    thrust::device_ptr<const T> rows_col_offset) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  rid = rows_in_bins[rid];  // permutation

  const auto global_col_offset = rows_col_offset[rid];
  const auto global_next_col_offset = rows_col_offset[rid + 1];

  T* hash_table = rows_col.get() + global_col_offset;
  const T table_sz = global_next_col_offset - global_col_offset;

  T nz = 0;

  for (T j = rpt_c[rid] + threadIdx.x; j < rpt_c[rid + 1]; j += blockDim.x) {
    T c_col = col_c[j];

    T hash = (c_col * 107) % table_sz;
    T offset = hash;

    while (true) {
      T table_value = hash_table[offset];
      if (table_value == c_col) {
        break;
      } else if (table_value == hash_invalidated) {
        T old_value = atomicCAS(hash_table + offset, hash_invalidated, c_col);
        if (old_value == hash_invalidated) {
          nz++;
          break;
        }
      } else {
        hash = (hash + 1) % table_sz;
        offset = hash;
      }
    }
  }

  for (T j = rpt_a[rid] + wid; j < rpt_a[rid + 1]; j += warpCount) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col] + i; k < rpt_b[a_col + 1]; k += warpSize) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % table_sz;
      T offset = hash;

      while (true) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            nz++;
            break;
          }
        } else {
          hash = (hash + 1) % table_sz;
          offset = hash;
        }
      }
    }
  }
}

template <typename T, unsigned int table_sz>
__global__ void fill_nz_block_row(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> rows_col,
    thrust::device_ptr<const T> rows_col_offset) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  __shared__ T hash_table[table_sz];

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  for (auto m = threadIdx.x; m < table_sz; m += blockDim.x) {
    hash_table[m] = hash_invalidated;
  }

  __syncthreads();

  rid = rows_in_bins[rid];  // permutation

  const auto global_col_offset = rows_col_offset[rid];

  T nz = 0;

  for (T j = rpt_c[rid] + threadIdx.x; j < rpt_c[rid + 1]; j += blockDim.x) {
    T c_col = col_c[j];

    T hash = (c_col * 107) % table_sz;
    T offset = hash;

    while (true) {
      T table_value = hash_table[offset];
      if (table_value == c_col) {
        break;
      } else if (table_value == hash_invalidated) {
        T old_value = atomicCAS(hash_table + offset, hash_invalidated, c_col);
        if (old_value == hash_invalidated) {
          nz++;
          break;
        }
      } else {
        hash = (hash + 1) % table_sz;
        offset = hash;
      }
    }
  }

  for (T j = rpt_a[rid] + wid; j < rpt_a[rid + 1]; j += warpCount) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col] + i; k < rpt_b[a_col + 1]; k += warpSize) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % table_sz;
      T offset = hash;

      while (true) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            nz++;
            break;
          }
        } else {
          hash = (hash + 1) % table_sz;
          offset = hash;
        }
      }
    }
  }

  __syncthreads();

  const T first_value = hash_table[0];

  if (threadIdx.x == 0) {
    if (first_value != hash_invalidated) {
      hash_table[0] = 1;
      rows_col[global_col_offset] = first_value;
    } else {
      hash_table[0] = 0;
    }
  }

  __syncthreads();

  T* stored = hash_table;

  for (auto j = threadIdx.x + 1; j < table_sz; j += blockDim.x) {
    auto val = hash_table[j];
    if (val != hash_invalidated) {
      auto index = atomicAdd(stored, 1);
      rows_col[global_col_offset + index] = val;
    }
  }
}

template <typename T, T pwarp, T block_sz, T max_per_row>
__global__ void fill_nz_pwarp_row(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> rows_col,
    thrust::device_ptr<const T> rows_col_offset, T n_rows) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  static_assert(block_sz % pwarp == 0);
  static_assert(block_sz >= pwarp);

  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ T hash_table[block_sz / pwarp * max_per_row];

  auto rid = tid / pwarp;
  auto i = tid % pwarp;
  auto local_rid = rid % (blockDim.x / pwarp);

  for (auto j = i; j < max_per_row; j += pwarp) {
    hash_table[local_rid * max_per_row + j] = hash_invalidated;
  }

  __syncwarp();

  if (rid >= n_rows)
    return;

  rid = rows_in_bins[rid];  // permutation

  const auto global_col_offset = rows_col_offset[rid];

  T nz = 0;

  for (T j = rpt_c[rid] + i; j < rpt_c[rid + 1]; j += pwarp) {
    T c_col = col_c[j];

    T hash = (c_col * 107) % max_per_row;
    T offset = hash + local_rid * max_per_row;

    while (true) {
      T table_value = hash_table[offset];
      if (table_value == c_col) {
        break;
      } else if (table_value == hash_invalidated) {
        T old_value = atomicCAS(hash_table + offset, hash_invalidated, c_col);
        if (old_value == hash_invalidated) {
          nz++;
          break;
        }
      } else {
        hash = (hash + 1) % max_per_row;
        offset = hash + local_rid * max_per_row;
      }
    }
  }

  for (T j = rpt_a[rid] + i; j < rpt_a[rid + 1]; j += pwarp) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col]; k < rpt_b[a_col + 1]; k++) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % max_per_row;
      T offset = hash + local_rid * max_per_row;

      while (true) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            nz++;
            break;
          }
        } else {
          hash = (hash + 1) % max_per_row;
          offset = hash + local_rid * max_per_row;
        }
      }
    }
  }

  auto mask = __activemask();
  __syncwarp(mask);

  T stored = 0;
  for (auto j = i; j < max_per_row; j += pwarp) {
    T val = hash_table[local_rid * max_per_row + j];
    auto store = val == hash_invalidated ? 0 : 1;

    for (auto shift = 1; shift <= pwarp / 2; shift *= 2) {
      auto other = __shfl_up_sync(mask, store, shift, pwarp);
      if (threadIdx.x % pwarp >= shift) {
        store += other;
      }
    }
    auto last_in_pwarp = __shfl_sync(mask, store, pwarp - 1, pwarp);

    if (val != hash_invalidated) {
      rows_col[global_col_offset + stored + store - 1] = val;
    }

    stored += last_in_pwarp;
  }
}

}  // namespace nsparse
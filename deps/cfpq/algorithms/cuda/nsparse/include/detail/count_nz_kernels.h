#pragma once

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename T>
__global__ void count_nz_block_row_large_global(
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> nz_per_row,
    thrust::device_ptr<T> global_hash_table,
    thrust::device_ptr<const T> hash_table_offset,
    thrust::device_ptr<const T> hash_table_size) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  T *hash_table = global_hash_table.get() + hash_table_offset[rid];
  T table_sz = hash_table_size[rid];
  __shared__ T snz;

  for (auto m = threadIdx.x; m < table_sz; m += blockDim.x) {
    hash_table[m] = hash_invalidated;
  }

  if (threadIdx.x == 0) {
    snz = 0;
  }

  __syncthreads();

  rid = rows_in_bins[rid]; // permutation
  T nz = 0;
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

  for (auto j = warpSize / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(0xffffffff, nz, j);
  }

  if (i == 0) {
    atomicAdd(&snz, nz);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    nz_per_row[rid] = snz;
  }
}

template <typename T, unsigned int table_sz>
__global__ void count_nz_block_row_large(
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> nz_per_row,
    thrust::device_ptr<T> fail_row_count, thrust::device_ptr<T> fail_row) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  __shared__ T hash_table[table_sz];
  __shared__ T nz;

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  for (auto m = threadIdx.x; m < table_sz; m += blockDim.x) {
    hash_table[m] = hash_invalidated;
  }

  if (threadIdx.x == 0) {
    nz = 0;
  }

  __syncthreads();

  T fail_count = 0;
  rid = rows_in_bins[rid]; // permutation
  for (T j = rpt_a[rid] + wid; j < rpt_a[rid + 1]; j += warpCount) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col] + i; k < rpt_b[a_col + 1]; k += warpSize) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % table_sz;
      T offset = hash;

      while (fail_count < table_sz / 2 && nz < table_sz / 2) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            atomicAdd(&nz, 1);
            break;
          }
        } else {
          hash = (hash + 1) % table_sz;
          offset = hash;
          fail_count++;
        }
      }
      if (fail_count >= table_sz / 2 || nz >= table_sz / 2) {
        break;
      }
    }
    if (fail_count >= table_sz / 2 || nz >= table_sz / 2) {
      break;
    }
  }

  __syncthreads();

  if (fail_count >= table_sz / 2 || nz >= table_sz / 2) {
    if (threadIdx.x == 0) {
      auto index = atomicAdd(fail_row_count.get(), 1);
      fail_row[index] = rid;
    }
  } else {
    if (threadIdx.x == 0) {
      nz_per_row[rid] = nz;
    }
  }
}

template <typename T, unsigned int table_sz>
__global__ void count_nz_block_row(thrust::device_ptr<const T> rpt_a,
                                   thrust::device_ptr<const T> col_a,
                                   thrust::device_ptr<const T> rpt_b,
                                   thrust::device_ptr<const T> col_b,
                                   thrust::device_ptr<const T> rows_in_bins,
                                   thrust::device_ptr<T> nz_per_row) {
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

  rid = rows_in_bins[rid]; // permutation
  T nz = 0;
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

  for (auto j = warpSize / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(0xffffffff, nz, j);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    hash_table[0] = 0;
  }

  __syncthreads();

  if (i == 0) {
    atomicAdd(hash_table, nz);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    nz_per_row[rid] = hash_table[0];
  }
}

template <typename T>
__global__ void count_nz_pwarp_row(thrust::device_ptr<const T> rpt_a,
                                   thrust::device_ptr<const T> col_a,
                                   thrust::device_ptr<const T> rpt_b,
                                   thrust::device_ptr<const T> col_b,
                                   thrust::device_ptr<const T> rows_in_bins,
                                   thrust::device_ptr<T> nz_per_row, T n_rows) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ T hash_table[2048];

  auto rid = tid / 4;
  auto i = tid % 4;
  auto local_rid = rid % (blockDim.x / 4);

  for (auto j = i; j < 32; j += 4) {
    hash_table[local_rid * 32 + j] = hash_invalidated;
  }

  __syncwarp();

  if (rid >= n_rows)
    return;

  rid = rows_in_bins[rid]; // permutation
  T nz = 0;
  for (T j = rpt_a[rid] + i; j < rpt_a[rid + 1]; j += 4) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col]; k < rpt_b[a_col + 1]; k++) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % 32;
      T offset = hash + local_rid * 32;

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
          hash = (hash + 1) % 32;
          offset = hash + local_rid * 32;
        }
      }
    }
  }

  auto mask = __activemask();
  for (auto j = 4 / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(mask, nz, j);
  }

  if (i == 0) {
    nz_per_row[rid] = nz;
  }
}
} // namespace nsparse
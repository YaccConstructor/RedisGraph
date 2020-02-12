#pragma once
#include <detail/fill_nz_kernels.h>

#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <utility>

namespace nsparse {

// inline size_t div_round_up(size_t m, size_t n) { return ceil((double)m / n);
// }

template <typename index_type>
thrust::device_vector<index_type>
fill_nz_per_row(index_type n_rows,
                const thrust::device_vector<index_type> &c_col_idx,
                const thrust::device_vector<index_type> &c_row_idx,
                const thrust::device_vector<index_type> &a_col_idx,
                const thrust::device_vector<index_type> &a_row_idx,
                const thrust::device_vector<index_type> &b_col_idx,
                const thrust::device_vector<index_type> &b_row_idx,
                const thrust::device_vector<index_type> &row_idx) {

  constexpr size_t bin_count = 7;
  thrust::device_vector<index_type> bin_size(bin_count, 0);

  thrust::for_each(thrust::counting_iterator<index_type>(0),
                   thrust::counting_iterator<index_type>(n_rows),
                   [row_per_bin = bin_size.data(),
                    rpt = row_idx.data()] __device__(index_type tid) {
                     size_t prod = rpt[tid + 1] - rpt[tid];

                     if (prod > 8192)
                       atomicAdd(row_per_bin.get() + 0, 1);
                     else if (prod > 4096)
                       atomicAdd(row_per_bin.get() + 1, 1);
                     else if (prod > 2048)
                       atomicAdd(row_per_bin.get() + 2, 1);
                     else if (prod > 1024)
                       atomicAdd(row_per_bin.get() + 3, 1);
                     else if (prod > 512)
                       atomicAdd(row_per_bin.get() + 4, 1);
                     else if (prod > 32)
                       atomicAdd(row_per_bin.get() + 5, 1);
                     else if (prod > 0)
                       atomicAdd(row_per_bin.get() + 6, 1);
                   });

  thrust::device_vector<index_type> bin_offset(bin_count);
  thrust::exclusive_scan(bin_size.begin(), bin_size.end(), bin_offset.begin());

  thrust::fill(bin_size.begin(), bin_size.end(), 0);

  thrust::device_vector<index_type> permutation_buffer(n_rows);

  thrust::for_each(
      thrust::counting_iterator<index_type>(0),
      thrust::counting_iterator<index_type>(n_rows),
      [rpt = row_idx.data(), bin_offset = bin_offset.data(),
       bin_size = bin_size.data(),
       rows_in_bins = permutation_buffer.data()] __device__(index_type tid) {
        auto prod = rpt[tid + 1] - rpt[tid];

        int bin = -1;

        if (prod > 8192)
          bin = 0;
        else if (prod > 4096)
          bin = 1;
        else if (prod > 2048)
          bin = 2;
        else if (prod > 1024)
          bin = 3;
        else if (prod > 512)
          bin = 4;
        else if (prod > 32)
          bin = 5;
        else if (prod > 0)
          bin = 6;

        if (bin == -1)
          return;

        auto curr_bin_size = atomicAdd(bin_size.get() + bin, 1);
        rows_in_bins[bin_offset[bin] + curr_bin_size] = tid;
      });

  index_type values_count = row_idx.back();

  thrust::device_vector<index_type> col_idx(
      values_count, std::numeric_limits<index_type>::max());

  for (auto bin_num = 1; bin_num < bin_count; bin_num++) {
    if (bin_size[bin_num] == 0)
      continue;

    switch (bin_num) {
    case 0:
      fill_nz_block_row_global<index_type><<<(unsigned int)bin_size[0], 1024>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[0], col_idx.data(),
          row_idx.data());
      break;
    case 1:
      fill_nz_block_row<index_type, 8192><<<(unsigned int)bin_size[1], 1024>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[1], col_idx.data(),
          row_idx.data());
      break;
    case 2:
      fill_nz_block_row<index_type, 4096><<<(unsigned int)bin_size[2], 512>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[2], col_idx.data(),
          row_idx.data());
      break;
    case 3:
      fill_nz_block_row<index_type, 2048><<<(unsigned int)bin_size[3], 256>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[3], col_idx.data(),
          row_idx.data());
      break;
    case 4:
      fill_nz_block_row<index_type, 1024><<<(unsigned int)bin_size[4], 128>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[4], col_idx.data(),
          row_idx.data());
      break;
    case 5:
      fill_nz_block_row<index_type, 512><<<(unsigned int)bin_size[5], 64>>>(
          c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
          a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
          permutation_buffer.data() + bin_offset[5], col_idx.data(),
          row_idx.data());
      break;
    case 6:
      fill_nz_pwarp_row<index_type>
          <<<div_round_up(bin_size[6] * 4, 256), 256>>>(
              c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
              a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
              permutation_buffer.data() + bin_offset[6], col_idx.data(),
              row_idx.data(), bin_size[6]);
      break;
    }
  }

//  thrust::host_vector<index_type> row_idx_host = row_idx;
//
//  for (auto i = 0; i < n_rows; i++) {
//    thrust::sort(col_idx.begin() + row_idx_host[i],
//                 col_idx.begin() + row_idx_host[i + 1]);
//  }

  return std::move(col_idx);
}

} // namespace nsparse

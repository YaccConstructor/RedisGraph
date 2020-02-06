#pragma once
#include <detail/count_nz_kernels.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <utility>

namespace nsparse {

template <typename index_type>
thrust::device_vector<index_type>
calc_nz_per_row(index_type n_rows,
                const thrust::device_vector<index_type> &a_col_idx,
                const thrust::device_vector<index_type> &a_row_idx,
                const thrust::device_vector<index_type> &b_col_idx,
                const thrust::device_vector<index_type> &b_row_idx) {

  constexpr size_t bin_count = 7;

  thrust::device_vector<index_type> bin_size(bin_count, 0);
  thrust::device_vector<index_type> products_per_row(n_rows + 1, 0);

  thrust::for_each(thrust::counting_iterator<index_type>(0),
                   thrust::counting_iterator<index_type>(n_rows),
                   [rpt_a = a_row_idx.data(), col_a = a_col_idx.data(),
                    rpt_b = b_row_idx.data(), col_b = b_col_idx.data(),
                    prod_per_row = products_per_row.data(),
                    row_per_bin = bin_size.data()] __device__(index_type tid) {
                     size_t prod = 0;
                     for (size_t j = rpt_a[tid]; j < rpt_a[tid + 1]; j++) {
                       prod += rpt_b[col_a[j] + 1] - rpt_b[col_a[j]];
                     }

                     prod_per_row[tid] = prod;
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
      [prod_per_row = products_per_row.data(), bin_offset = bin_offset.data(),
       bin_size = bin_size.data(),
       rows_in_bins = permutation_buffer.data()] __device__(index_type tid) {
        auto prod = prod_per_row[tid];

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

  for (auto bin_num = 0; bin_num < bin_count; bin_num++) {
    if (bin_size[bin_num] == 0)
      continue;

    switch (bin_num) {
    case 0: {
      thrust::device_vector<index_type> fail_count(1, 0);
      thrust::device_vector<index_type> fail_row(bin_size[0]);
      count_nz_block_row_large<index_type, 8192>
          <<<(unsigned int)bin_size[0], 1024>>>(
              a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
              b_col_idx.data(), permutation_buffer.data() + bin_offset[0],
              products_per_row.data(), fail_count.data(), fail_row.data());
      index_type fail_cnt = fail_count[0];
      if (fail_cnt > 0) {
        thrust::device_vector<index_type> fail_row_product_per_row(fail_cnt);
        thrust::device_vector<index_type> fail_row_hash_table_offsets(fail_cnt);

        thrust::transform(fail_row.begin(), fail_row.begin() + fail_cnt,
                          fail_row_product_per_row.begin(),
                          [ptr = products_per_row.data()] __device__(
                              auto item) { return ptr[item]; });

        thrust::exclusive_scan(fail_row_product_per_row.begin(),
                               fail_row_product_per_row.end(),
                               fail_row_hash_table_offsets.begin());

        thrust::device_vector<index_type> hash_table(thrust::reduce(
            fail_row_product_per_row.begin(), fail_row_product_per_row.end()));

        count_nz_block_row_large_global<index_type><<<fail_cnt, 1024>>>(
            a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
            b_col_idx.data(), fail_row.data(), products_per_row.data(),
            hash_table.data(), fail_row_hash_table_offsets.data(),
            fail_row_product_per_row.data());
      }
    } break;
    case 1:
      count_nz_block_row<index_type, 8192><<<(unsigned int)bin_size[1], 1024>>>(
          a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
          b_col_idx.data(), permutation_buffer.data() + bin_offset[1],
          products_per_row.data());
      break;
    case 2:
      count_nz_block_row<index_type, 4096><<<(unsigned int)bin_size[2], 512>>>(
          a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
          b_col_idx.data(), permutation_buffer.data() + bin_offset[2],
          products_per_row.data());
      break;
    case 3:
      count_nz_block_row<index_type, 2048><<<(unsigned int)bin_size[3], 256>>>(
          a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
          b_col_idx.data(), permutation_buffer.data() + bin_offset[3],
          products_per_row.data());
      break;
    case 4:
      count_nz_block_row<index_type, 1024><<<(unsigned int)bin_size[4], 128>>>(
          a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
          b_col_idx.data(), permutation_buffer.data() + bin_offset[4],
          products_per_row.data());
      break;
    case 5:
      count_nz_block_row<index_type, 512><<<(unsigned int)bin_size[5], 64>>>(
          a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
          b_col_idx.data(), permutation_buffer.data() + bin_offset[5],
          products_per_row.data());
      break;
    case 6:
      count_nz_pwarp_row<index_type>
          <<<div_round_up(bin_size[6] * 4, 256), 256>>>(
              a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
              b_col_idx.data(), permutation_buffer.data() + bin_offset[6],
              products_per_row.data(), bin_size[6]);
      break;
    }
  }

  thrust::exclusive_scan(products_per_row.begin(), products_per_row.end(),
                         products_per_row.begin());

  return std::move(products_per_row);
}

} // namespace nsparse
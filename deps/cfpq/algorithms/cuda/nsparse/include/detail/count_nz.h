#pragma once
#include <detail/meta.h>
#include <detail/util.h>
#include <detail/count_nz_kernels.h>

#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <utility>
#include <vector>

namespace nsparse {

template <typename index_type>
struct global_hash_table_state_t {
  thrust::device_vector<index_type> hash_table;
  thrust::device_vector<index_type> hashed_row_offsets;
  thrust::device_vector<index_type> hashed_row_indices;
};

template <typename index_type>
struct row_index_res_t {
  thrust::device_vector<index_type> row_index;
  global_hash_table_state_t<index_type> global_hash_table_state;
};

template <typename index_type, typename... Borders>
void exec_pwarp_row(const thrust::device_vector<index_type>& c_col_idx,
                    const thrust::device_vector<index_type>& c_row_idx,
                    const thrust::device_vector<index_type>& a_col_idx,
                    const thrust::device_vector<index_type>& a_row_idx,
                    const thrust::device_vector<index_type>& b_col_idx,
                    const thrust::device_vector<index_type>& b_row_idx,
                    const thrust::device_vector<index_type>& permutation_buffer,
                    const thrust::device_vector<index_type>& bin_offset,
                    const thrust::device_vector<index_type>& bin_size,
                    thrust::device_vector<index_type>& row_idx, std::tuple<Borders...>) {
  constexpr index_type pwarp = 4;
  constexpr index_type block_sz = 256;

  EXPAND_SIDE_EFFECTS(
      (bin_size[Borders::bin_index] > 0
           ? count_nz_pwarp_row<index_type, pwarp, block_sz, Borders::max_border>
           <<<util::div(bin_size[Borders::bin_index] * pwarp, block_sz), block_sz>>>(
               c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
               b_row_idx.data(), b_col_idx.data(),
               permutation_buffer.data() + bin_offset[Borders::bin_index], row_idx.data(),
               bin_size[Borders::bin_index])
           : void()));
}

template <typename index_type, typename... Borders>
void exec_block_row(const thrust::device_vector<index_type>& c_col_idx,
                    const thrust::device_vector<index_type>& c_row_idx,
                    const thrust::device_vector<index_type>& a_col_idx,
                    const thrust::device_vector<index_type>& a_row_idx,
                    const thrust::device_vector<index_type>& b_col_idx,
                    const thrust::device_vector<index_type>& b_row_idx,
                    const thrust::device_vector<index_type>& permutation_buffer,
                    const thrust::device_vector<index_type>& bin_offset,
                    const thrust::device_vector<index_type>& bin_size,
                    thrust::device_vector<index_type>& row_idx, std::tuple<Borders...>) {
  static_assert(meta::all_of<(Borders::max_border / 8 % 32 == 0)...>);

  EXPAND_SIDE_EFFECTS(
      (bin_size[Borders::bin_index] > 0 ? count_nz_block_row<index_type, Borders::max_border>
           <<<(index_type)bin_size[Borders::bin_index], Borders::max_border / 8>>>(
               c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
               b_row_idx.data(), b_col_idx.data(),
               permutation_buffer.data() + bin_offset[Borders::bin_index], row_idx.data())
                                        : void()));
}

template <typename index_type, typename Border>
global_hash_table_state_t<index_type> exec_global_row(
    const thrust::device_vector<index_type>& c_col_idx,
    const thrust::device_vector<index_type>& c_row_idx,
    const thrust::device_vector<index_type>& a_col_idx,
    const thrust::device_vector<index_type>& a_row_idx,
    const thrust::device_vector<index_type>& b_col_idx,
    const thrust::device_vector<index_type>& b_row_idx,
    const thrust::device_vector<index_type>& permutation_buffer,
    const thrust::device_vector<index_type>& bin_offset,
    const thrust::device_vector<index_type>& bin_size, thrust::device_vector<index_type>& row_idx,
    std::tuple<Border>) {
  index_type size = bin_size[Border::bin_index];

  if (size == 0)
    return {};

  constexpr index_type block_sz = 1024;

  static_assert(block_sz % 32 == 0);

  // extra item for counter in last position
  thrust::device_vector<index_type> fail_stat(size + 1, 0);

  count_nz_block_row_large<index_type, Border::min_border><<<size, block_sz>>>(
      c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
      b_col_idx.data(), permutation_buffer.data() + bin_offset[Border::bin_index], row_idx.data(),
      fail_stat.data() + size, fail_stat.data());

  index_type fail_count = fail_stat.back();
  if (fail_count > 0) {
    fail_stat.resize(fail_count);
    thrust::device_vector<index_type> hash_table_offsets(fail_count + 1);

    thrust::transform(
        fail_stat.begin(), fail_stat.end(), hash_table_offsets.begin(),
        [ptr = row_idx.data()] __device__(index_type item) { return ptr[item] * 1.1; });
    hash_table_offsets.back() = 0;

    thrust::exclusive_scan(hash_table_offsets.begin(), hash_table_offsets.end(),
                           hash_table_offsets.begin());

    index_type hash_table_sz = hash_table_offsets.back();
    thrust::device_vector<index_type> hash_table(hash_table_sz);

    count_nz_block_row_large_global<index_type><<<fail_count, block_sz>>>(
        c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
        b_col_idx.data(), fail_stat.data(), row_idx.data(), hash_table.data(),
        hash_table_offsets.data());

    return {std::move(hash_table), std::move(hash_table_offsets), std::move(fail_stat)};
  }
  return {};
}

template <typename index_type, typename... Borders>
row_index_res_t<index_type> calc_nz_per_row(index_type n_rows, index_type n_cols,
                                            const thrust::device_vector<index_type>& c_col_idx,
                                            const thrust::device_vector<index_type>& c_row_idx,
                                            const thrust::device_vector<index_type>& a_col_idx,
                                            const thrust::device_vector<index_type>& a_row_idx,
                                            const thrust::device_vector<index_type>& b_col_idx,
                                            const thrust::device_vector<index_type>& b_row_idx,
                                            std::tuple<Borders...>) {
  constexpr size_t bin_count = sizeof...(Borders);
  constexpr size_t unused_bin = meta::max_bin<Borders...> + 1;

  thrust::device_vector<index_type> bin_size(bin_count, 0);
  thrust::device_vector<index_type> products_per_row(n_rows + 1, 0);

  thrust::for_each(
      thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n_rows),
      [rpt_a = a_row_idx.data(), col_a = a_col_idx.data(), rpt_b = b_row_idx.data(),
       col_b = b_col_idx.data(), rpt_c = c_row_idx.data(), row_per_bin = bin_size.data(),
       max_c_cols = n_cols, prod_per_row = products_per_row.data()] __device__(index_type tid) {
        index_type prod = 0;
        for (size_t j = rpt_a[tid]; j < rpt_a[tid + 1]; j++) {
          prod += rpt_b[col_a[j] + 1] - rpt_b[col_a[j]];
        }

        prod += (rpt_c[tid + 1] - rpt_c[tid]);

        prod = min(prod, max_c_cols);

        prod_per_row[tid] = prod;

        size_t bin = meta::select_bin<Borders...>(prod, unused_bin);
        if (bin != unused_bin)
          atomicAdd(row_per_bin.get() + bin, 1);
      });

  thrust::device_vector<index_type> bin_offset(bin_count);
  thrust::exclusive_scan(bin_size.begin(), bin_size.end(), bin_offset.begin());

  thrust::fill(bin_size.begin(), bin_size.end(), 0);

  thrust::device_vector<index_type> permutation_buffer(n_rows);

  thrust::for_each(thrust::counting_iterator<index_type>(0),
                   thrust::counting_iterator<index_type>(n_rows),
                   [prod_per_row = products_per_row.data(), bin_offset = bin_offset.data(),
                    bin_size = bin_size.data(),
                    rows_in_bins = permutation_buffer.data()] __device__(index_type tid) {
                     auto prod = prod_per_row[tid];

                     int bin = meta::select_bin<Borders...>(prod, unused_bin);

                     if (bin == unused_bin)
                       return;

                     auto curr_bin_size = atomicAdd(bin_size.get() + bin, 1);
                     rows_in_bins[bin_offset[bin] + curr_bin_size] = tid;
                   });

  exec_pwarp_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                 permutation_buffer, bin_offset, bin_size, products_per_row,
                 meta::filter<meta::pwarp_row, Borders...>);

  exec_block_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                 permutation_buffer, bin_offset, bin_size, products_per_row,
                 meta::filter<meta::block_row, Borders...>);

  auto global_hash_table_state = exec_global_row(
      c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx, permutation_buffer,
      bin_offset, bin_size, products_per_row, meta::filter<meta::global_row, Borders...>);

  thrust::exclusive_scan(products_per_row.begin(), products_per_row.end(),
                         products_per_row.begin());

  return {std::move(products_per_row), std::move(global_hash_table_state)};
}

}  // namespace nsparse
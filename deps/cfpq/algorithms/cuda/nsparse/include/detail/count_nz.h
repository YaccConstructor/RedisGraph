#pragma once
#include <detail/meta.h>
#include <detail/util.h>
#include <detail/count_nz.cuh>

#include <cub/cub.cuh>

#include <iostream>

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
  constexpr size_t pwarp = 4;

  EXPAND_SIDE_EFFECTS(
      (bin_size[Borders::bin_index] > 0
           ? count_nz_pwarp_row<index_type, pwarp, Borders::config_t::block_size,
                                Borders::max_border>
           <<<util::div(bin_size[Borders::bin_index] * pwarp, Borders::config_t::block_size),
              Borders::config_t::block_size>>>(
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
  static_assert(meta::all_of<(Borders::config_t::block_size % 32 == 0)...>);

  EXPAND_SIDE_EFFECTS(
      (bin_size[Borders::bin_index] > 0 ? count_nz_block_row<index_type, Borders::max_border>
           <<<(index_type)bin_size[Borders::bin_index], Borders::config_t::block_size>>>(
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

  constexpr size_t table_size = Border::config_t::block_size * 8;

  if (size == 0)
    return {};

  thrust::device_vector<index_type> aka_fail_stat(
      permutation_buffer.begin() + bin_offset[Border::bin_index],
      permutation_buffer.begin() + bin_offset[Border::bin_index] + size);

  thrust::device_vector<index_type> bucket_count(size + 1);
  bucket_count.back() = 0;

  thrust::transform(permutation_buffer.begin() + bin_offset[Border::bin_index],
                    permutation_buffer.begin() + bin_offset[Border::bin_index] + size,
                    bucket_count.begin(), [prod = row_idx.data()] __device__(auto row_id) {
                      index_type p = prod[row_id];
                      // add extra bucket for decrease collision count
                      return (p + table_size - 1) / table_size + 1;
                    });

  using namespace util;

  thrust::exclusive_scan(bucket_count.begin(), bucket_count.end(), bucket_count.begin());
  index_type total_buckets = bucket_count.back();

  thrust::device_vector<util::bucket_info_t<index_type>> bucket_info(total_buckets);

  thrust::for_each(
      thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(size),
      [rpt_a = a_row_idx.data(), rpt_b = b_row_idx.data(), col_a = a_col_idx.data(),
       bucket_ptr = bucket_info.data(), bucket_cnt = bucket_count.data(),
       rows_ids = aka_fail_stat.data(), prod_in = row_idx.data()] __device__(index_type item) {
        index_type prod = 0;
        index_type offset = bucket_cnt[item];
        index_type total_buckets = bucket_cnt[item + 1] - offset;
        index_type row_id = rows_ids[item];
        index_type divide = (prod_in[row_id] + total_buckets - 1) / total_buckets;
        index_type part_count = 0;

        index_type j;
        index_type k;
        index_type prev_k;
        index_type prev_j;
        for (j = rpt_a[row_id]; j < rpt_a[row_id + 1]; j++) {
          for (k = rpt_b[col_a[j]]; k < rpt_b[col_a[j] + 1]; k++) {
            if (prod % divide == 0) {
              if (part_count != 0) {
                // update prev
                (bucket_ptr.get() + offset - 1)->a_row_end = prev_j;
                (bucket_ptr.get() + offset - 1)->b_row_end = prev_k;
              }
              bucket_ptr[offset++] = util::bucket_info_t<index_type>{row_id, j, k, 0, 0};
              part_count++;
            }
            prev_k = k + 1;
            prev_j = j + 1;
            prod++;
          }
        }

        (bucket_ptr.get() + offset - 1)->a_row_end = j;
        (bucket_ptr.get() + offset - 1)->b_row_end = k;
      });

  thrust::for_each(aka_fail_stat.begin(), aka_fail_stat.end(),
                   [prod = row_idx.data()] __device__(index_type row_id) { prod[row_id] = 0; });

  thrust::device_vector<index_type> hash_table(total_buckets * table_size);

  count_nz_block_row_large<index_type, table_size><<<total_buckets, Border::config_t::block_size>>>(
      c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(), b_row_idx.data(),
      b_col_idx.data(), bucket_info.data(), hash_table.data(), row_idx.data());

  thrust::device_vector<index_type> hash_table_offsets(size + 1);

  thrust::transform(bucket_count.begin(), bucket_count.end(), hash_table_offsets.begin(),
                    [] __device__(index_type it) { return it * table_size; });

  thrust::device_vector<index_type> sorted_hash_table(hash_table.size());
  {
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_storage_bytes, thrust::raw_pointer_cast(hash_table.data()),
        thrust::raw_pointer_cast(sorted_hash_table.data()), hash_table.size(), size,
        thrust::raw_pointer_cast(hash_table_offsets.data()),
        thrust::raw_pointer_cast(hash_table_offsets.data()) + 1);

    thrust::device_vector<char> storage(temp_storage_bytes);

    cub::DeviceSegmentedRadixSort::SortKeys(
        thrust::raw_pointer_cast(storage.data()), temp_storage_bytes,
        thrust::raw_pointer_cast(hash_table.data()),
        thrust::raw_pointer_cast(sorted_hash_table.data()), hash_table.size(), size,
        thrust::raw_pointer_cast(hash_table_offsets.data()),
        thrust::raw_pointer_cast(hash_table_offsets.data()) + 1);
  }

  cudaDeviceSynchronize();

  return {std::move(sorted_hash_table), std::move(hash_table_offsets), std::move(aka_fail_stat)};
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

        //        prod = min(prod, max_c_cols);

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
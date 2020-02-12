#pragma once
#include <cassert>
#include <matrix.h>

#include <thrust/iterator/counting_iterator.h>

#include <detail/count_nz.h>
#include <detail/fill_nz.h>

namespace nsparse {

template <typename index_type>
const matrix<bool, index_type> spgemm(const matrix<bool, index_type> &c,
                                      const matrix<bool, index_type> &a,
                                      const matrix<bool, index_type> &b) {
  assert(a.m_n_cols == b.m_n_rows);
  assert(c.m_n_rows == a.m_n_rows);
  assert(c.m_n_cols == b.m_n_cols);

  index_type n_rows = a.m_n_rows;
  index_type n_cols = b.m_n_cols;

  nz_per_row_res_t<index_type> res = calc_nz_per_row(
      n_rows, n_cols, c.m_col_index, c.m_row_index, a.m_col_index,
      a.m_row_index, b.m_col_index, b.m_row_index);

  thrust::device_vector<index_type> col_index = fill_nz_per_row(
      n_rows, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
      b.m_col_index, b.m_row_index, res.row_index);

  if (res.rows_in_table.size() > 0) {
    filter_hash_table<index_type><<<res.rows_in_table.size(), 1>>>(
        res.row_index.data(), res.hash_table.data(),
        res.hash_table_offsets.data(), res.hash_table_sizes.data(),
        res.rows_in_table.data(), col_index.data());
  }

//  thrust::host_vector<index_type> row_index_host = res.row_index;
//
//  for (auto i = 0; i < n_rows; i++) {
//    thrust::sort(col_index.begin() + row_index_host[i],
//                 col_index.begin() + row_index_host[i + 1]);
//  }

  //  for (auto i = 0; i < res.rows_in_table.size(); i++) {
  //    index_type size = res.hash_table_sizes[i];
  //    index_type offset = res.hash_table_offsets[i];
  //    index_type row_id = res.rows_in_table[i];
  //    thrust::copy_if(
  //        res.hash_table.begin() + offset, res.hash_table.begin() + offset +
  //        size, col_index.begin() + res.row_index[row_id], [] __device__(auto
  //        item) {
  //          return item != std::numeric_limits<index_type>::max();
  //        });
  //  }

  assert(res.row_index.size() == n_rows + 1);
  assert(col_index.size() == res.row_index.back());
  return {col_index, std::move(res.row_index), n_rows, n_cols};
}

} // namespace nsparse
#pragma once
#include <cassert>
#include <matrix.h>

#include <thrust/iterator/counting_iterator.h>

#include <detail/count_nz.h>
#include <detail/fill_nz.h>

namespace nsparse {

template <typename index_type>
const matrix<bool, index_type> spgemm(const matrix<bool, index_type> &a,
                                      const matrix<bool, index_type> &b) {
  assert(a.m_n_cols == b.m_n_rows);

  index_type n_rows = a.m_n_rows;
  index_type n_cols = b.m_n_cols;

  thrust::device_vector<index_type> row_index = calc_nz_per_row(
      n_rows, a.m_col_index, a.m_row_index, b.m_col_index, b.m_row_index);

  thrust::device_vector<index_type> col_index =
      fill_nz_per_row(n_rows, a.m_col_index, a.m_row_index, b.m_col_index,
                      b.m_row_index, row_index);

  assert(row_index.size() == n_rows + 1);
  return {col_index, row_index, n_rows, n_cols};
}

} // namespace nsparse
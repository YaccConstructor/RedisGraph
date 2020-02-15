#pragma once
#include <cassert>
#include <matrix.h>

#include <thrust/iterator/counting_iterator.h>

#include <detail/count_nz.h>
#include <detail/fill_nz.h>

namespace nsparse {

template <typename index_type>
const matrix<bool, index_type> spgemm(const matrix<bool, index_type>& c,
                                      const matrix<bool, index_type>& a,
                                      const matrix<bool, index_type>& b) {
  assert(a.m_cols == b.m_rows);
  assert(c.m_rows == a.m_rows);
  assert(c.m_cols == b.m_cols);

  index_type rows = a.m_rows;
  index_type cols = b.m_cols;

  using namespace meta;
  constexpr auto config_find_nz =
      std::tuple<bin_info_t<8192, std::numeric_limits<size_t>::max(), 0, global_row>,
                 bin_info_t<4096, 8192, 1, block_row>, bin_info_t<2048, 4096, 2, block_row>,
                 bin_info_t<1024, 2048, 3, block_row>, bin_info_t<512, 1024, 4, block_row>,
                 bin_info_t<32, 512, 5, block_row>, bin_info_t<0, 32, 6, pwarp_row>>{};

  row_index_res_t<index_type> res =
      calc_nz_per_row(rows, cols, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                      b.m_col_index, b.m_row_index, config_find_nz);

  constexpr auto config_fill_nz =
      std::tuple<bin_info_t<4096, 8192, 0, block_row>, bin_info_t<2048, 4096, 1, block_row>,
                 bin_info_t<1024, 2048, 2, block_row>, bin_info_t<512, 1024, 3, block_row>,
                 bin_info_t<32, 512, 4, block_row>, bin_info_t<0, 32, 5, pwarp_row>>{};

  thrust::device_vector<index_type> col_index =
      fill_nz_per_row(rows, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                      b.m_col_index, b.m_row_index, res.row_index, config_fill_nz);

  reuse_global_hash_table(res.row_index, col_index, res.global_hash_table_state);

  assert(res.row_index.size() == rows + 1);
  assert(col_index.size() == res.row_index.back());
  index_type vals = col_index.size();
  return {std::move(col_index), std::move(res.row_index), rows, cols, vals};
}

}  // namespace nsparse
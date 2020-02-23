#pragma once
#include <cassert>
#include <matrix.h>

#include <detail/merge_path.cuh>

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
      std::tuple<bin_info_t<1024, std::numeric_limits<size_t>::max(), 0, global_row>,
                 bin_info_t<512, 1024, 1, block_row>, bin_info_t<32, 512, 2, block_row>,
                 bin_info_t<0, 32, 3, pwarp_row>>{};

  row_index_res_t<index_type> res =
      calc_nz_per_row(rows, cols, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                      b.m_col_index, b.m_row_index, config_find_nz);

  constexpr auto config_fill_nz =
      std::tuple<bin_info_t<512, 1024, 0, block_row>, bin_info_t<32, 512, 1, block_row>,
                 bin_info_t<0, 32, 2, pwarp_row>>{};

  thrust::device_vector<index_type> col_index =
      fill_nz_per_row(rows, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                      b.m_col_index, b.m_row_index, res.row_index, config_fill_nz);

  reuse_global_hash_table(res.row_index, col_index, res.global_hash_table_state);

  validate_order<index_type><<<rows, 128>>>(res.row_index.data(), col_index.data());
  validate_order<index_type><<<rows, 128>>>(c.m_row_index.data(), c.m_col_index.data());

  thrust::device_vector<index_type> rpt_result(rows + 1, 0);
  merge_count<index_type><<<rows, 32>>>(res.row_index.data(), col_index.data(),
                                        c.m_row_index.data(), c.m_col_index.data(),
                                        rpt_result.data());

  {
    thrust::device_vector<index_type> lol_rpt(rows + 1, 0);
    //    thrust::for_each(thrust::counting_iterator<index_type>(0),
    //                     thrust::counting_iterator<index_type>(rows + 1),
    //                     [one = res.row_index.data(), two = c.m_row_index.data(),
    //                      lol = lol_rpt.data()] __device__(index_type i) { lol[i] = one[i] +
    //                      two[i]; });
    //
    thrust::device_vector<index_type> lol_result;

    merge_path<index_type, 256><<<rows, 128>>>(res.row_index.data(), col_index.data(),
                                               c.m_row_index.data(), c.m_col_index.data(),
                                               lol_rpt.data(), lol_result.data());

//    rpt_result = std::move(lol_rpt);
  }

  thrust::exclusive_scan(rpt_result.begin(), rpt_result.end(), rpt_result.begin());

  thrust::device_vector<index_type> col_result(rpt_result.back());
  merge<index_type><<<rows, 32>>>(res.row_index.data(), col_index.data(), c.m_row_index.data(),
                                  c.m_col_index.data(), rpt_result.data(), col_result.data());

  assert(rpt_result.size() == rows + 1);
  assert(col_result.size() == rpt_result.back());
  index_type vals = col_result.size();
  return {std::move(col_result), std::move(rpt_result), rows, cols, vals};
}

}  // namespace nsparse
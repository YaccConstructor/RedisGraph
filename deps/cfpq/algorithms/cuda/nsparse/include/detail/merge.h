#pragma once

#include <detail/merge_path.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace nsparse {

template <typename index_type>
std::pair<thrust::device_vector<index_type>, thrust::device_vector<index_type>> unique_merge(
    const thrust::device_vector<index_type>& rpt_a, const thrust::device_vector<index_type>& col_a,
    const thrust::device_vector<index_type>& rpt_b,
    const thrust::device_vector<index_type>& col_b) {
  assert(rpt_a.size() == rpt_b.size());
  assert(rpt_a.size() > 0);

  auto rows = rpt_a.size() - 1;

  thrust::device_vector<index_type> rpt_c(rows + 1, 0);

  merge_path_count<index_type, 128>
      <<<rows, 128>>>(rpt_a.data(), col_a.data(), rpt_b.data(), col_b.data(), rpt_c.data());

  thrust::exclusive_scan(rpt_c.begin(), rpt_c.end(), rpt_c.begin());

  thrust::device_vector<index_type> col_c(rpt_c.back());
  merge_path_fill<index_type, 128><<<rows, 128>>>(rpt_a.data(), col_a.data(), rpt_b.data(),
                                                  col_b.data(), rpt_c.data(), col_c.data());

  return {std::move(rpt_c), std::move(col_c)};
}

}  // namespace nsparse
#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace nsparse {

template <typename ValueType, typename IndexType> class matrix;

template <typename IndexType> class matrix<bool, IndexType> {
public:
  typedef IndexType index_type;
  typedef bool value_type;

  matrix() : m_col_index{}, m_row_index{}, m_n_rows{0}, m_n_cols{0} {}

  matrix(index_type n_rows, index_type n_cols)
      : m_col_index{}, m_row_index{n_rows + 1, 0}, m_n_rows{n_rows}, m_n_cols{n_cols} {}

  matrix(thrust::device_vector<index_type> col_index,
         thrust::device_vector<index_type> row_index, index_type n_rows,
         index_type n_cols)
      : m_col_index{std::move(col_index)},
        m_row_index{std::move(row_index)}, m_n_rows{n_rows}, m_n_cols{n_cols} {}

  thrust::device_vector<index_type> m_col_index;
  thrust::device_vector<index_type> m_row_index;
  index_type m_n_rows;
  index_type m_n_cols;
};

} // namespace nsparse
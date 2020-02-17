#include <gtest/gtest.h>
#include <cub/cub.cuh>

#include <mult.h>

using b_mat = std::vector<std::vector<bool>>;

std::pair<std::vector<unsigned int>, std::vector<unsigned int>> dense_to_csr(const b_mat& matrix);

b_mat matrix_generator(size_t rows, size_t cols, float density);

b_mat mult(const b_mat& a, const b_mat& b);

b_mat sum(const b_mat& a, const b_mat& b);

template <typename T>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T> vec) {
  for (auto i = 0; i < vec.size(); i++) {
    os << vec[i] << " ";
  }
  return os;
}

class NsparseCountNonZeroTest : public testing::Test {
 protected:
  static void eval(const b_mat& c, const b_mat& a, const b_mat& b) {
    b_mat r = sum(c, mult(a, b));

    auto sprsA = dense_to_csr(a);
    auto sprsB = dense_to_csr(b);
    auto sprsC = dense_to_csr(c);
    auto sprsR = dense_to_csr(r);

    nsparse::matrix<bool, unsigned int> A(sprsA.first, sprsA.second, a.size(), a[0].size(),
                                          sprsA.second.back());
    nsparse::matrix<bool, unsigned int> B(sprsB.first, sprsB.second, b.size(), b[0].size(),
                                          sprsB.second.back());
    nsparse::matrix<bool, unsigned int> C(sprsC.first, sprsC.second, c.size(), c[0].size(),
                                          sprsC.second.back());

    auto res = nsparse::spgemm<>(C, A, B);

    ASSERT_EQ(sprsR.second, res.m_row_index);

    {
      thrust::device_vector<unsigned int> new_col(res.m_col_index.size());

      size_t temp_storage_bytes = 0;
      cub::DeviceSegmentedRadixSort::SortKeys(
          nullptr, temp_storage_bytes, thrust::raw_pointer_cast(res.m_col_index.data()),
          thrust::raw_pointer_cast(new_col.data()), res.m_col_index.size(),
          res.m_row_index.size() - 1, thrust::raw_pointer_cast(res.m_row_index.data()),
          thrust::raw_pointer_cast(res.m_row_index.data()) + 1);

      thrust::device_vector<char> storage(temp_storage_bytes);

      cub::DeviceSegmentedRadixSort::SortKeys(
          thrust::raw_pointer_cast(storage.data()), temp_storage_bytes,
          thrust::raw_pointer_cast(res.m_col_index.data()),
          thrust::raw_pointer_cast(new_col.data()), res.m_col_index.size(),
          res.m_row_index.size() - 1, thrust::raw_pointer_cast(res.m_row_index.data()),
          thrust::raw_pointer_cast(res.m_row_index.data()) + 1);

      res.m_col_index = std::move(new_col);
    }

    ASSERT_EQ(sprsR.first, res.m_col_index);
  }
};

TEST_F(NsparseCountNonZeroTest, countNzSmall) {
  eval(
      {
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
      },
      {
          {0, 1, 0, 0, 1, 0},
          {1, 0, 1, 0, 1, 0},
          {0, 0, 0, 0, 0, 0},
          {0, 1, 1, 0, 0, 0},
      },
      {
          {0, 0, 1, 0, 0},
          {1, 0, 1, 0, 1},
          {1, 1, 1, 1, 1},
          {0, 0, 0, 0, 0},
          {0, 1, 0, 0, 0},
          {0, 0, 1, 1, 1},
      });
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedSmall) {
  size_t a = 100;
  size_t b = 150;
  size_t c = 200;

  for (float density = 0.01; density <= 1; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedMedium) {
  size_t a = 500;
  size_t b = 600;
  size_t c = 700;

  for (float density = 0.01; density <= 0.5; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedBig) {
  size_t a = 1000;
  size_t b = 1100;
  size_t c = 1200;

  for (float density = 0.01; density <= 0.2; density += 0.01) {
    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
         matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedGlobalHashTable) {
  size_t a = 100;
  size_t b = 500;
  size_t c = 5000;

  eval(matrix_generator(a, c, 0.5), matrix_generator(a, b, 0.5), matrix_generator(b, c, 0.5));
}
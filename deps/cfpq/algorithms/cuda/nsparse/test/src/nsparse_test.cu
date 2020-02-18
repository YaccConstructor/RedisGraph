#include <gtest/gtest.h>
#include <cub/cub.cuh>
#include <detail/merge_path.cuh>
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

TEST_F(NsparseCountNonZeroTest, mergeTest) {
  std::vector<uint> col_a = {1, 3, 4, 6, 15, 16, 17, 40, 41, 42, 43};
  std::vector<uint> rpt_a = {0, 11};

  std::vector<uint> col_b = {2,  3,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  std::vector<uint> rpt_b = {0, 27};

  thrust::device_vector<uint> rpt_c(rpt_a.size(), 0);

  nsparse::merge_count<uint><<<1, 32>>>(thrust::device_vector<uint>(rpt_a).data(),
                                        thrust::device_vector<uint>(col_a).data(),
                                        thrust::device_vector<uint>(rpt_b).data(),
                                        thrust::device_vector<uint>(col_b).data(), rpt_c.data());

  thrust::exclusive_scan(rpt_c.begin(), rpt_c.end(), rpt_c.begin());
  ASSERT_EQ(std::vector<uint>({0, 33}), rpt_c);

  thrust::device_vector<uint> col_c(rpt_c.back());

  nsparse::merge<uint><<<1, 32>>>(
      thrust::device_vector<uint>(rpt_a).data(), thrust::device_vector<uint>(col_a).data(),
      thrust::device_vector<uint>(rpt_b).data(), thrust::device_vector<uint>(col_b).data(),
      rpt_c.data(), col_c.data());

  ASSERT_EQ(std::vector<uint>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43}),
            col_c);
}

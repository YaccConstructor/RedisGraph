#include <gtest/gtest.h>
#include <cub/cub.cuh>
#include <detail/merge_path.cuh>
#include <mult.h>
#include <random>

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

// TEST_F(NsparseCountNonZeroTest, countNzSmall) {
//  eval(
//      {
//          {0, 0, 0, 0, 0},
//          {0, 0, 0, 0, 0},
//          {0, 0, 0, 0, 0},
//          {0, 0, 0, 0, 0},
//      },
//      {
//          {0, 1, 0, 0, 1, 0},
//          {1, 0, 1, 0, 1, 0},
//          {0, 0, 0, 0, 0, 0},
//          {0, 1, 1, 0, 0, 0},
//      },
//      {
//          {0, 0, 1, 0, 0},
//          {1, 0, 1, 0, 1},
//          {1, 1, 1, 1, 1},
//          {0, 0, 0, 0, 0},
//          {0, 1, 0, 0, 0},
//          {0, 0, 1, 1, 1},
//      });
//}
//
// TEST_F(NsparseCountNonZeroTest, countNzGeneratedSmall) {
//  size_t a = 100;
//  size_t b = 150;
//  size_t c = 200;
//
//  for (float density = 0.01; density <= 1; density += 0.01) {
//    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
//         matrix_generator(b, c, density));
//  }
//}
//
// TEST_F(NsparseCountNonZeroTest, countNzGeneratedMedium) {
//  size_t a = 500;
//  size_t b = 600;
//  size_t c = 700;
//
//  for (float density = 0.01; density <= 0.5; density += 0.01) {
//    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
//         matrix_generator(b, c, density));
//  }
//}
//
// TEST_F(NsparseCountNonZeroTest, countNzGeneratedBig) {
//  size_t a = 1000;
//  size_t b = 1100;
//  size_t c = 1200;
//
//  for (float density = 0.01; density <= 0.2; density += 0.01) {
//    eval(matrix_generator(a, c, density), matrix_generator(a, b, density),
//         matrix_generator(b, c, density));
//  }
//}
//
// TEST_F(NsparseCountNonZeroTest, countNzGeneratedGlobalHashTable) {
//  size_t a = 100;
//  size_t b = 500;
//  size_t c = 5000;
//
//  eval(matrix_generator(a, c, 0.5), matrix_generator(a, b, 0.5), matrix_generator(b, c, 0.5));
//}

std::vector<int> merge_path(const std::vector<int>& a, const std::vector<int>& b) {
  constexpr int block = 5;

  constexpr int min = std::numeric_limits<int>::min();
  constexpr int max = std::numeric_limits<int>::max();

  size_t block_count = (a.size() + b.size() + block - 1) / block;

  size_t begin_a = 0;
  size_t begin_b = 0;

  std::vector<int> res;

  for (auto i = 0; i < block_count; i++) {
    assert(begin_a <= a.size());
    assert(begin_b <= b.size());

    // load in buf
    int buf_a_size = 0;
    int buf_b_size = 0;

    int raw_a[block + 2];
    for (auto& item : raw_a) {
      item = max;
    }
    int raw_b[block + 2];
    for (auto& item : raw_b) {
      item = max;
    }

    raw_a[0] = max;
    raw_b[0] = max;

    int* buffer_a = raw_a + 1;
    int* buffer_b = raw_b + 1;

    for (auto j = 0; j < block; j++) {
      if (j + begin_a < a.size()) {
        buf_a_size++;
        buffer_a[j] = a[j + begin_a];
      }
    }

    for (auto j = 0; j < block; j++) {
      if (j + begin_b < b.size()) {
        buf_b_size++;
        buffer_b[j] = b[j + begin_b];
      }
    }

    int max_x_index = 0;
    int max_y_index = 0;

    for (auto j = 0; j < std::min(buf_b_size + buf_a_size, block); j++) {
      const int ind_sum = j + 2;
      const int y = ind_sum;
      const int x = 0;
      const int iter_count = ind_sum;
      assert(iter_count > 0);

      int l = 0;
      int r = iter_count;

      while (r - l > 1) {
        int step = (r - l) / 2;
        int check = l + step;

        int check_x = x + check;
        int check_y = y - check;

        bool ans = raw_b[check_y] > raw_a[check_x];

        l += step * ans;
        r -= step * !ans;
      }

      int ans_x = x + l;
      int ans_y = y - l;

      //      if (ans_x == 0 || raw_b[ans_y - 1] > raw_a[ans_x]) {
      //        res.push_back(raw_b[ans_y - 1]);
      //        max_y_index = std::max(max_y_index, ans_y - 1);
      //      } else {
      //        res.push_back(raw_a[ans_x]);
      //        max_x_index = std::max(max_x_index, ans_x);
      //      }

      if (ans_y == 1) {
        res.push_back(raw_a[ans_x]);
        max_x_index = std::max(max_x_index, ans_x);
      } else if (ans_x == 0) {
        res.push_back(raw_b[ans_y - 1]);
        max_y_index = std::max(max_y_index, ans_y - 1);
      } else if (raw_b[ans_y - 1] > raw_a[ans_x]) {
        res.push_back(raw_b[ans_y - 1]);
        max_y_index = std::max(max_y_index, ans_y - 1);
      } else {
        res.push_back(raw_a[ans_x]);
        max_x_index = std::max(max_x_index, ans_x);
      }
    }

    begin_a += max_x_index;
    begin_b += max_y_index;
  }

  return res;
}

TEST_F(NsparseCountNonZeroTest, mergePathTest) {
  std::mt19937 gen;
  std::uniform_int_distribution<> ud(0, 1000000);
  for (auto sz_a = 0; sz_a < 5000; sz_a += 12) {
    for (auto sz_b = 0; sz_b < 5000; sz_b += 13) {
      std::vector<int> a;
      for (auto i = 0; i < sz_a; i++) {
        a.push_back(ud(gen));
      }
      std::vector<int> b;
      for (auto i = 0; i < sz_b; i++) {
        b.push_back(ud(gen));
      }

      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());

      std::vector<int> expected(sz_a + sz_b);
      std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin());

      {
        thrust::device_vector<int> rpt_a(2);
        rpt_a[0] = 0;
        rpt_a[1] = sz_a;

        thrust::device_vector<int> col_a(sz_a);
        col_a = a;

        thrust::device_vector<int> rpt_b(2);
        rpt_b[0] = 0;
        rpt_b[1] = sz_b;

        thrust::device_vector<int> col_b(sz_b);
        col_b = b;

        thrust::device_vector<int> rpt_c(2);
        rpt_c[0] = 0;
        rpt_c[1] = sz_a + sz_b;

        thrust::device_vector<int> col_c(sz_a + sz_b);

        nsparse::merge_path<int, 256><<<1, 128>>>(rpt_a.data(), col_a.data(), rpt_b.data(), col_b.data(),
                                             rpt_c.data(), col_c.data());

        ASSERT_EQ(expected, col_c);
      }

      ASSERT_EQ(expected, merge_path(a, b));
    }
  }

  std::vector<int> a = {1, 3, 4, 6, 15, 16, 17, 40, 41, 42, 43};
  std::vector<int> b = {0, 22, 100, 111};

  std::vector<int> res(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), res.begin());

  ASSERT_EQ(res, merge_path(a, b));
}
//
// TEST_F(NsparseCountNonZeroTest, mergeTest) {
//  std::vector<uint> col_a = {1, 3, 4, 6, 15, 16, 17, 40, 41, 42, 43};
//  std::vector<uint> rpt_a = {0, 11};
//
//  std::vector<uint> col_b = {2,  3,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
//                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
//  std::vector<uint> rpt_b = {0, 27};
//
//  thrust::device_vector<uint> rpt_c(rpt_a.size(), 0);
//
//  nsparse::merge_count<uint><<<1, 32>>>(thrust::device_vector<uint>(rpt_a).data(),
//                                        thrust::device_vector<uint>(col_a).data(),
//                                        thrust::device_vector<uint>(rpt_b).data(),
//                                        thrust::device_vector<uint>(col_b).data(), rpt_c.data());
//
//  thrust::exclusive_scan(rpt_c.begin(), rpt_c.end(), rpt_c.begin());
//  ASSERT_EQ(std::vector<uint>({0, 33}), rpt_c);
//
//  thrust::device_vector<uint> col_c(rpt_c.back());
//
//  nsparse::merge<uint><<<1, 32>>>(
//      thrust::device_vector<uint>(rpt_a).data(), thrust::device_vector<uint>(col_a).data(),
//      thrust::device_vector<uint>(rpt_b).data(), thrust::device_vector<uint>(col_b).data(),
//      rpt_c.data(), col_c.data());
//
//  ASSERT_EQ(std::vector<uint>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
//                               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43}),
//            col_c);
//}

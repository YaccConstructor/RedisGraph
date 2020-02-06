#include <detail/count_nz.h>
#include <detail/fill_nz_kernels.h>
#include <gtest/gtest.h>
#include <thrust/transform.h>

#include <mult.h>

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void foo(thrust::device_ptr<int> in, thrust::device_ptr<int> out) {
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;

  int val = in[tid];

  for (int i = 1; i <= 1; i *= 2) {
    auto tmp = __shfl_up_sync(0xffffffff, val, i, 2);

    //    if ((threadIdx.x % 4) >= i)
    val += tmp;
  }

  out[tid] = val;
}

void bar() {
  thrust::device_vector<int> a = std::vector<unsigned int>{
      1, 2, 3, 4, 12, 4, 7, 1, 8, 4, 7, 5, 0, 0, 5, 1,
      1, 2, 3, 4, 12, 4, 7, 1, 8, 4, 7, 5, 0, 0, 5, 100};

  thrust::device_vector<int> b(32);

  foo<<<1, 32>>>(a.data(), b.data());

  cudaDeviceSynchronize();

  for (int i = 0; i < 32; i++) {
    std::cout << b[i] << " ";
  }
}

using b_mat = std::vector<std::vector<bool>>;

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
dense_to_csr(const b_mat &matrix);

b_mat matrix_generator(size_t rows, size_t cols, float density);

b_mat mult(const b_mat &a, const b_mat &b);

template <typename T>
std::ostream &operator<<(std::ostream &os, const thrust::device_vector<T> vec) {
  for (auto i = 0; i < vec.size(); i++) {
    os << vec[i] << " ";
  }
  return os;
}

class NsparseCountNonZeroTest : public testing::Test {

protected:
  static void eval(const b_mat &a, const b_mat &b) {
    b_mat c = mult(a, b);

    auto sprsA = dense_to_csr(a);
    auto sprsB = dense_to_csr(b);
    auto sprsC = dense_to_csr(c);

    std::vector<unsigned int> nz_per_row(c.size());

    nsparse::matrix<bool, unsigned int> A(sprsA.first, sprsA.second, a.size(),
                                          a[0].size());
    nsparse::matrix<bool, unsigned int> B(sprsB.first, sprsB.second, b.size(),
                                          b[0].size());

    auto res = nsparse::spgemm<>(A, B);

    ASSERT_EQ(sprsC.second, res.m_row_index);
    ASSERT_EQ(sprsC.first, res.m_col_index);
  }
};

TEST_F(NsparseCountNonZeroTest, countNzSmall) {
  //  bar();
  eval(
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
    eval(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedMedium) {
  size_t a = 500;
  size_t b = 600;
  size_t c = 700;

  for (float density = 0.01; density <= 0.2; density += 0.01) {
    eval(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedBig) {
  size_t a = 1000;
  size_t b = 1100;
  size_t c = 1200;

  for (float density = 0.01; density <= 0.1; density += 0.01) {
    eval(matrix_generator(a, b, density), matrix_generator(b, c, density));
  }
}

TEST_F(NsparseCountNonZeroTest, countNzGeneratedGlobalHashTable) {
  size_t a = 100;
  size_t b = 500;
  size_t c = 5000;

  eval(matrix_generator(a, b, 0.5), matrix_generator(b, c, 0.5));
}
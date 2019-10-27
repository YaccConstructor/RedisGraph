#pragma once

#include "Constants.h"
#include "Matrix.h"
#include "gpuMemoryManagement.h"
#include "methodOf4RusBooleanSemiringGpu.h"
#include <vector>

class MethodOf4RusMatricesEnv : public MatricesEnv {
public:
  int size_multiple_by_32;
  int cols;
  TYPE *extra_matrix_device;
  gpu_m4ri::Tables tables;

  void
  environment_preprocessing(const std::vector<Matrix *> &matrices) override;

  void
  environment_postprocessing(const std::vector<Matrix *> &matrices) override;
};

class MethodOf4RusMatrix : public Matrix {
public:
  int size_multiple_by_32;
  int cols;
  TYPE *matrix_host;
  TYPE *matrix_device;
  MethodOf4RusMatricesEnv *env;

  explicit MethodOf4RusMatrix(unsigned int n);

  ~MethodOf4RusMatrix() override;

  MethodOf4RusMatrix(MethodOf4RusMatrix &&) noexcept;
  MethodOf4RusMatrix &operator=(MethodOf4RusMatrix &&) noexcept;

  MethodOf4RusMatrix(const MethodOf4RusMatrix &) = delete;
  MethodOf4RusMatrix &operator=(const MethodOf4RusMatrix &) = delete;

  void set_bit(unsigned int row, unsigned col) override;

  unsigned int get_bit(unsigned int row, unsigned col) override;

  bool add_mul(Matrix *left, Matrix *right) override;
  size_t bit_count() override;
};

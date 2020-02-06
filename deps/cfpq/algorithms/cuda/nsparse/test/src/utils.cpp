#include <random>
#include <vector>

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
dense_to_csr(const std::vector<std::vector<bool>> &matrix) {
  std::vector<unsigned int> col_index;
  std::vector<unsigned int> row_index;

  unsigned int size = 0;
  for (const auto &row : matrix) {
    row_index.push_back(size);
    for (unsigned int i = 0; i < row.size(); i++) {
      if (row[i]) {
        col_index.push_back(i);
        size++;
      }
    }
  }
  row_index.push_back(size);

  return {col_index, row_index};
}

std::vector<std::vector<bool>> matrix_generator(size_t rows, size_t cols,
                                                float density) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> urd(0, 1);

  std::vector<std::vector<bool>> matrix(rows);

  for (size_t i = 0; i < rows; i++) {
    matrix[i].reserve(cols);
    for (size_t j = 0; j < cols; j++) {
      matrix[i].push_back(urd(gen) <= density ? 1 : 0);
    }
  }

  return matrix;
}

std::vector<std::vector<bool>> mult(const std::vector<std::vector<bool>> &a,
                                    const std::vector<std::vector<bool>> &b) {

  const auto row = a.size();
  const auto col = b[0].size();
  const auto mid = a[0].size();

  std::vector<std::vector<bool>> res(row);

  for (auto i = 0; i < row; i++) {
    res[i].reserve(col);
    for (auto j = 0; j < col; j++) {
      bool val = false;
      for (auto k = 0; k < mid; k++) {
        val = val || (a[i][k] && b[k][j]);
      }
      res[i].push_back(val);
    }
  }

  return res;
}
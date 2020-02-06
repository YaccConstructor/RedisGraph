#include "triplet_loader.h"

#include <istream>
#include <memory>
#include <vector>

std::map<std::string, GrB_Matrix> load_triplets(std::istream &istream) {
  GrB_Index from, to;
  std::string label;

  std::map<std::string,
           std::pair<std::vector<GrB_Index>, std::vector<GrB_Index>>>
      parsed;

  GrB_Index max_dimension = 0;

  while (istream >> from) {
    istream >> label;
    istream >> to;
    auto &indices = parsed[label];
    indices.first.push_back(from);
    indices.second.push_back(to);

    max_dimension = std::max(max_dimension, std::max(from, to));
  }

  max_dimension += 1;

  std::map<std::string, GrB_Matrix> result;

  for (const auto &[name, data] : parsed) {
    GrB_Matrix matrix;
    size_t nvals = data.first.size();

    GrB_Matrix_new(&matrix, GrB_BOOL, max_dimension, max_dimension);

    std::unique_ptr<bool[]> values(new bool[nvals]{true});

    GrB_Matrix_build_BOOL(matrix, data.first.data(), data.second.data(),
                          values.get(), nvals, GrB_FIRST_BOOL);

    result[name] = matrix;
  }

  return result;
}

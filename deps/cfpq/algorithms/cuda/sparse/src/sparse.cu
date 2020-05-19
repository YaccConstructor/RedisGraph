#include <sparse.h>
#include <generic.hpp>
#include <thrust/detail/config/compiler_fence.h>

#include <cusp/coo_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/multiply.h>

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/mr/allocator.h>
#include <thrust/memory/detail/device_system_resource.h>

struct PathIndex {
  __host__ __device__ PathIndex() = default;
  __host__ __device__  PathIndex(int) {
    left = -1;
    right = -1;
    middle = -1;
    height = -1;
  }

  GrB_Index left;
  GrB_Index right;
  GrB_Index middle;
  GrB_Index height;
};

bool __device__ operator==(const PathIndex& lhs, const PathIndex& rhs) {
  return lhs.left == rhs.left && lhs.right == rhs.right && lhs.middle == rhs.middle &&
         lhs.height == rhs.height;
}

template <typename T>
struct id {
  using result_type = T;
  __device__ __host__ T operator()(const T& v) {
    return v;
  }
};

struct mult {
  using result_type = PathIndex;
  __device__ __host__ PathIndex operator()(const PathIndex& lhs, const PathIndex& rhs) {
    PathIndex res{};
    res.left = lhs.left;
    res.right = rhs.right;
    res.middle = lhs.right;
    res.height = max(lhs.height, rhs.height) + 1;
    return res;
  }
};

struct add {
  using result_type = PathIndex;
  __device__ __host__ PathIndex operator()(const PathIndex& lhs, const PathIndex& rhs) {
    if (lhs.height < rhs.height) {
      return lhs;
    }
    return rhs;
  }
};

namespace {
struct matrix {
  explicit matrix(const GrB_Matrix& other) {
    GrB_Matrix tmp_matrix;
    GrB_Matrix_dup(&tmp_matrix, other);

    GrB_Type tp;
    GrB_Index nrows, ncols, nvals;
    int64_t nonempty;

    GrB_Index* col_idx;
    GrB_Index* row_idx;
    void* vals;

    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);

    GxB_Matrix_export_CSR(&tmp_matrix, &tp, &nrows, &ncols, &nvals, &nonempty, &row_idx, &col_idx,
                          &vals, desc);

    using IndexArray = cusp::array1d<GrB_Index, cusp::host_memory>;
    using ValueArray = cusp::array1d<PathIndex, cusp::host_memory>;

    IndexArray col(col_idx, col_idx + nvals);
    IndexArray row(row_idx, row_idx + nrows + 1);
    ValueArray val(nvals);

    cusp::csr_matrix_view<IndexArray::view, IndexArray::view, ValueArray::view> view(
        nrows, ncols, nvals, row, col, val);
    data_ = view;
  }

  explicit matrix(size_t sz) : data_{sz, sz, 0} {
  }

  explicit matrix(cusp::coo_matrix<GrB_Index, PathIndex, cusp::device_memory> data)
      : data_(std::move(data)) {
  }

  matrix(matrix&&) = default;
  matrix& operator=(matrix&&) = default;

  size_t vals() const {
    return data_.num_entries;
  }

  cusp::coo_matrix<GrB_Index, PathIndex, cusp::device_memory> data_;
};

struct functor {
  matrix operator()(const matrix& d, const matrix& a, const matrix& b) {
    id<PathIndex> initialize;
    mult combine{};
    add reduce{};


    cusp::coo_matrix<GrB_Index, PathIndex, cusp::device_memory> c{};
    cusp::multiply(a.data_, b.data_, c, initialize, combine, reduce);

    cusp::coo_matrix<GrB_Index, PathIndex, cusp::device_memory> res{};
    cusp::elementwise(c, d.data_, res, reduce);
    return matrix{std::move(res)};
  }
};
}  // namespace

int sparse(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
           const char** relations_names, size_t relations_count, size_t graph_size) {
  auto matrices = algorithms::matrix_init<matrix>(grammar, response, relations, relations_names,
                                                  relations_count, graph_size);
  auto res = algorithms::matrix_closure<matrix, functor>(grammar, matrices);
  algorithms::fill_response<matrix>(grammar, matrices, response, res.first);
  return 0;
}

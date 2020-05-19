#include <cassert>
#include "cpu_graphblas.h"
#include "generic.hpp"

void check_info(GrB_Info info) {
  assert(info == GrB_SUCCESS);
}

typedef struct {
  uint32_t left;
  uint32_t right;
  uint32_t middle;

  uint32_t height;
  uint32_t length;
} PathIndex;

PathIndex PathIndex_Identity = {
    .left = 0,
    .right = 0,
    .middle = 0,
    .height = 0,
    .length = 0,
};

void PathIndex_Init(PathIndex* index, uint32_t left, uint32_t right, uint32_t middle,
                    uint32_t height, uint32_t length) {
  index->left = left;
  index->right = right;
  index->middle = middle;
  index->height = height;
  index->length = length;
}

void PathIndex_InitIdentity(PathIndex* index) {
  index->left = 0;
  index->right = 0;
  index->middle = 0;
  index->height = 0;
  index->length = 0;
}

bool PathIndex_IsIdentity(const PathIndex* index) {
  return index->left == 0 && index->right == 0 && index->middle == 0 && index->height == 0 &&
         index->length == 0;
}

bool PathIndex_IsEdge(const PathIndex* index) {
  return index->length == 1;
}

void PathIndex_Copy(const PathIndex* from, PathIndex* to) {
  to->left = from->left;
  to->right = from->right;
  to->middle = from->middle;
  to->height = from->height;
  to->length = from->length;
}

void PathIndex_Mul(void* z, const void* x, const void* y) {
  PathIndex* left = (PathIndex*)x;
  PathIndex* right = (PathIndex*)y;
  PathIndex* res = (PathIndex*)z;

  if (!PathIndex_IsIdentity(left) && !PathIndex_IsIdentity(right)) {
    uint32_t height = (left->height < right->height ? right->height : left->height) + 1;
    PathIndex_Init(res, left->left, right->right, left->right, height,
                   left->length + right->length);

  } else {
    PathIndex_InitIdentity(res);
  }
}

void PathIndex_Add(void* z, const void* x, const void* y) {
  const PathIndex* left = (const PathIndex*)x;
  const PathIndex* right = (const PathIndex*)y;
  PathIndex* res = (PathIndex*)z;

  if (!PathIndex_IsIdentity(left) && !PathIndex_IsIdentity(right)) {
    const PathIndex* min_height_index = (left->height < right->height) ? left : right;
    PathIndex_Copy(min_height_index, res);
  } else if (PathIndex_IsIdentity(left)) {
    PathIndex_Copy(right, res);
  } else {
    PathIndex_Copy(left, res);
  }
}

void PathIndex_MatrixInit(GrB_Matrix* m, const GrB_Matrix* bool_m) {
  GrB_Index nvals;
  GrB_Matrix_nvals(&nvals, *bool_m);

  GrB_Index* I = new GrB_Index[nvals];
  GrB_Index* J = new GrB_Index[nvals];
  bool* X = new bool[nvals];

  GrB_Matrix_extractTuples_BOOL(I, J, X, &nvals, *bool_m);
  for (int k = 0; k < nvals; ++k) {
    PathIndex index;
    PathIndex_Init(&index, I[k], J[k], 0, 1, 1);
    GrB_Matrix_setElement_UDT(*m, (void*)&index, I[k], J[k]);
  }

  delete[] I;
  delete[] J;
  delete[] X;
}

namespace {

GrB_Type IndexType_;

GrB_Type GetIndexType() {
  static bool registered = false;
  if (!registered) {
    check_info(GrB_Type_new(&IndexType_, sizeof(PathIndex)));
    registered = true;
  }
  return IndexType_;
}

struct matrix {
  matrix() noexcept : handle(nullptr) {
  }

  explicit matrix(GrB_Matrix& other) : handle(nullptr) {
    GrB_Index ncols, nrows;
    GrB_Matrix_ncols(&ncols, other);
    GrB_Matrix_ncols(&nrows, other);

    GrB_Matrix_new(&handle, GetIndexType(), nrows, ncols);

    PathIndex_MatrixInit(&handle, &other);
  }

  matrix(matrix&& other) noexcept {
    handle = other.handle;
    other.handle = nullptr;
  }

  matrix& operator=(matrix&& other) noexcept {
    std::swap(other.handle, handle);
    return *this;
  }

  ~matrix() {
    GrB_Matrix_free(&handle);
  }

  size_t vals() const {
    size_t v = 0;
    GrB_Matrix_nvals(&v, handle);
    return v;
  }

  GrB_Matrix handle;
};

struct functor {
  functor() {
    check_info(GrB_BinaryOp_new(&IndexType_Add, PathIndex_Add, GetIndexType(), GetIndexType(),
                                GetIndexType()));
    check_info(GrB_BinaryOp_new(&IndexType_Mul, PathIndex_Mul, GetIndexType(), GetIndexType(),
                                GetIndexType()));
    check_info(GrB_Monoid_new_UDT(&IndexType_Monoid, IndexType_Add, (void*)&PathIndex_Identity));
    check_info(GrB_Semiring_new(&IndexType_Semiring, IndexType_Monoid, IndexType_Mul));
  }

  matrix operator()(matrix& c, const matrix& a, const matrix& b) {
    GrB_Info info = GrB_mxm(c.handle, GrB_NULL, IndexType_Add, IndexType_Semiring, a.handle,
                            b.handle, GrB_NULL);
    assert(info == GrB_SUCCESS);
    return std::move(c);
  }

  GrB_BinaryOp IndexType_Add, IndexType_Mul;

  GrB_Monoid IndexType_Monoid;

  GrB_Semiring IndexType_Semiring;

  ~functor() {
  }
};

}  // namespace
int cpu_graphblas(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                  const char** relations_names, size_t relations_count, size_t graph_size) {
  auto matrices = algorithms::matrix_init<matrix>(grammar, response, relations, relations_names,
                                                  relations_count, graph_size);
  auto res = algorithms::matrix_closure<matrix, functor>(grammar, matrices);
  algorithms::fill_response<matrix>(grammar, matrices, response, res.first);
  return 0;
}

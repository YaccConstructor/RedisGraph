#include "cusparse_cfpq.h"

#include <thrust/device_vector.h>
#include <cusparse.h>
#include <generic.hpp>

namespace {
struct matrix {
  matrix() = default;

  matrix(thrust::device_vector<int> row_, thrust::device_vector<int> col_)
      : row(std::move(row_)), col(std::move(col_)) {
  }

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

    thrust::device_vector<GrB_Index> col_host(col_idx, col_idx + nvals);
    thrust::device_vector<GrB_Index> row_host(row_idx, row_idx + nrows + 1);

    col = thrust::device_vector<int>(col_host);
    row = thrust::device_vector<int>(row_host);
  }

  explicit matrix(size_t sz) : row{sz + 1, 0}, col() {
  }

  int vals() const {
    return row.back();
  }

  matrix(matrix&& other) = default;
  matrix& operator=(matrix&& other) = default;

  thrust::device_vector<int> row;
  thrust::device_vector<int> col;
};

struct functor {
  functor() : handle(nullptr) {
    cusparseCreate(&handle);
  }

  matrix operator()(const matrix& d, const matrix& a, const matrix& b) {
    const auto graph_size = a.row.size() - 1;

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descrB;
    cusparseCreateMatDescr(&descrB);
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descrD;
    cusparseCreateMatDescr(&descrD);
    cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descrC;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

    int nnzC = 0;
    csrgemm2Info_t info = NULL;
    size_t bufferSize = 0;
    // nnzTotalDevHostPtr points to host memory
    int* nnzTotalDevHostPtr = &nnzC;
    double alpha = 1;
    double beta = 1;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // C = A * B + D

    // step 1: create an opaque structure
    cusparseCreateCsrgemm2Info(&info);

    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    cusparseDcsrgemm2_bufferSizeExt(handle, graph_size, graph_size, graph_size, &alpha, descrA,
                                    a.col.size(), a.row.data().get(), a.col.data().get(), descrB,
                                    b.col.size(), b.row.data().get(), b.col.data().get(), &beta,
                                    descrD, d.col.size(), d.row.data().get(), d.col.data().get(),
                                    info, &bufferSize);

    thrust::device_vector<char> buffer(bufferSize);

    // step 3: compute csrRowPtrC

    thrust::device_vector<int> row(graph_size + 1);
    cusparseXcsrgemm2Nnz(handle, graph_size, graph_size, graph_size, descrA, a.col.size(),
                         a.row.data().get(), a.col.data().get(), descrB, b.col.size(),
                         b.row.data().get(), b.col.data().get(), descrD, d.col.size(),
                         d.row.data().get(), d.col.data().get(), descrC, row.data().get(),
                         nnzTotalDevHostPtr, info, buffer.data().get());
    assert(nnzTotalDevHostPtr != NULL);
    nnzC = *nnzTotalDevHostPtr;

    thrust::device_vector<int> col(nnzC);

    cusparseDcsrgemm2(handle, graph_size, graph_size, graph_size, &alpha, descrA, a.col.size(),
                      NULL, a.row.data().get(), a.col.data().get(), descrB, b.col.size(), NULL,
                      b.row.data().get(), b.col.data().get(), &beta, descrD, d.col.size(), NULL,
                      d.row.data().get(), d.col.data().get(), descrC, NULL, row.data().get(),
                      col.data().get(), info, buffer.data().get());

    cusparseDestroyCsrgemm2Info(info);

    return matrix(std::move(row), std::move(col));
  }

  cusparseHandle_t handle;
};
}  // namespace

int cusparse_cfpq(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                  const char** relations_names, size_t relations_count, size_t graph_size) {
  auto matrices = algorithms::matrix_init<matrix>(grammar, response, relations, relations_names,
                                                  relations_count, graph_size);
  auto res = algorithms::matrix_closure<matrix, functor>(grammar, matrices);
  algorithms::fill_response<matrix>(grammar, matrices, response, res.first);
  return 0;
}
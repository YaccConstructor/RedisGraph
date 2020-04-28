#include "cusparse_cfpq.h"

#include <item_mapper.h>
#include <thrust/device_vector.h>
#include <cusparse.h>

#include <chrono>


using namespace std::chrono;

struct matrix {
  thrust::device_vector<int> row;
  thrust::device_vector<int> col;
};

int cusparse_cfpq(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                  const char** relations_names, size_t relations_count, size_t graph_size) {
  size_t nonterm_count = grammar->nontermMapper.count;

  auto t1 = high_resolution_clock::now();

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  std::vector<matrix> matrices(
      nonterm_count, {thrust::device_vector<int>{graph_size + 1, 0}, thrust::device_vector<int>{}});

  // Initialize matrices
  for (int i = 0; i < relations_count; i++) {
    const char* terminal = relations_names[i];

    MapperIndex terminal_id =
        ItemMapper_GetPlaceIndex((ItemMapper*)&grammar->tokenMapper, terminal);
    if (terminal_id != grammar->tokenMapper.count) {
      for (int j = 0; j < grammar->simple_rules_count; j++) {
        const SimpleRule* simpleRule = &grammar->simple_rules[j];
        if (simpleRule->r == terminal_id) {
          GrB_Matrix tmp_matrix;
          GrB_Matrix_dup(&tmp_matrix, relations[i]);

          GrB_Type tp;
          GrB_Index nrows, ncols, nvals;
          int64_t nonempty;

          GrB_Index* col_idx;
          GrB_Index* row_idx;
          void* vals;

          GrB_Descriptor desc;
          GrB_Descriptor_new(&desc);

          GxB_Matrix_export_CSR(&tmp_matrix, &tp, &nrows, &ncols, &nvals, &nonempty, &row_idx,
                                &col_idx, &vals, desc);

          thrust::device_vector<int> col_index(col_idx, col_idx + nvals);
          thrust::device_vector<int> row_index(row_idx, row_idx + nrows + 1);

          matrices[simpleRule->l] = {std::move(row_index), std::move(col_index)};
        }
      }
    }
  }

  auto t2 = high_resolution_clock::now();
  response->time_to_prepare += duration<double, seconds::period>(t2 - t1).count();


  bool matrices_is_changed = true;
  while (matrices_is_changed) {
    matrices_is_changed = false;

    response->iteration_count++;
    for (int i = 0; i < grammar->complex_rules_count; ++i) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      assert(graph_size == matrices[nonterm1].row.size() - 1);

      auto before = matrices[nonterm1].col.size();

      {
        cusparseMatDescr_t descrA;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        matrix& a_data = matrices[nonterm2];

        cusparseMatDescr_t descrB;
        cusparseCreateMatDescr(&descrB);
        cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
        matrix& b_data = matrices[nonterm3];

        cusparseMatDescr_t descrD;
        cusparseCreateMatDescr(&descrD);
        cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ZERO);
        matrix& d_data = matrices[nonterm1];

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
        cusparseDcsrgemm2_bufferSizeExt(
            handle, graph_size, graph_size, graph_size, &alpha, descrA, a_data.col.size(),
            a_data.row.data().get(), a_data.col.data().get(), descrB, b_data.col.size(),
            b_data.row.data().get(), b_data.col.data().get(), &beta, descrD, d_data.col.size(),
            d_data.row.data().get(), d_data.col.data().get(), info, &bufferSize);

          thrust::device_vector<char> buffer(bufferSize);

        // step 3: compute csrRowPtrC

        matrix C;
        C.row = thrust::device_vector<int>(graph_size + 1);
        cusparseXcsrgemm2Nnz(handle, graph_size, graph_size, graph_size, descrA, a_data.col.size(),
                             a_data.row.data().get(), a_data.col.data().get(), descrB,
                             b_data.col.size(), b_data.row.data().get(), b_data.col.data().get(),
                             descrD, d_data.col.size(), d_data.row.data().get(),
                             d_data.col.data().get(), descrC, C.row.data().get(),
                             nnzTotalDevHostPtr, info, buffer.data().get());
        assert(nnzTotalDevHostPtr != NULL);
        nnzC = *nnzTotalDevHostPtr;

        C.col = thrust::device_vector<int>(nnzC);

        cusparseDcsrgemm2(
            handle, graph_size, graph_size, graph_size, &alpha, descrA, a_data.col.size(),
            NULL /*a_data.val.data().get()*/, a_data.row.data().get(), a_data.col.data().get(),
            descrB, b_data.col.size(), NULL /*b_data.val.data().get()*/, b_data.row.data().get(),
            b_data.col.data().get(), &beta, descrD, d_data.col.size(),
            NULL /*d_data.val.data().get()*/, d_data.row.data().get(), d_data.col.data().get(),
            descrC, NULL /*C.val.data().get()*/, C.row.data().get(), C.col.data().get(), info,
            buffer.data().get());

        cusparseDestroyCsrgemm2Info(info);

        cudaDeviceSynchronize();

        matrices[nonterm1] = std::move(C);
      }

      auto after = matrices[nonterm1].col.size();

      matrices_is_changed |= (before != after);
    }
  }

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char* nonterm;

    nvals = matrices[i].col.size();
    nonterm = ItemMapper_Map((ItemMapper*)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}
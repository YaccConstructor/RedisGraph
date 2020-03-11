#include "matrix.h"
#include "spgemm.h"
#include "nsparse.h"
#include "masked_spgemm.h"

#include <chrono>
#include <grammar.h>
#include <item_mapper.h>
#include <map>
#include <response.h>
#include <set>
#include <vector>
#include <iostream>
#include <masked_matrix.h>

using namespace std::chrono;

template <typename value_type, typename index_type>
void index_path(std::vector<nsparse::matrix<bool, index_type>>& init_matrices,
                const std::vector<nsparse::matrix<bool, index_type>>& final_matrices,
                const std::vector<std::tuple<int, int, int>>& evaluation_plan,
                index_type graph_size, index_type nonterm_count) {
  std::vector<nsparse::masked_matrix<value_type, index_type>> masked_matrices;
  masked_matrices.reserve(nonterm_count);

  constexpr value_type zero = std::numeric_limits<value_type>::max();

  {
    value_type edge = 1;
    edge <<= sizeof(value_type) * 8 / 2;
    edge += std::numeric_limits<index_type>::max();

    auto identity = nsparse::masked_matrix<value_type, index_type>::identity(graph_size, 0);
    auto id_mul = [] __device__(value_type lhs, value_type rhs, index_type a_col) -> value_type {
      return lhs;
    };
    auto id_add = [] __device__(value_type * lhs, value_type rhs) -> void { *lhs = rhs; };

    nsparse::masked_spgemm_functor_t<value_type, index_type, zero, decltype(id_mul),
                                     decltype(id_add)>
        masked_id_spgemm(id_mul, id_add);

    for (auto i = 0; i < nonterm_count; i++) {
      masked_matrices.emplace_back(final_matrices[i], -1);

      index_type left_size = init_matrices[i].m_vals;

      nsparse::masked_matrix<value_type, index_type> left(
          std::move(init_matrices[i]), thrust::device_vector<value_type>(left_size, edge));

      masked_id_spgemm(masked_matrices.back(), left, identity);
      cudaDeviceSynchronize();
    }
  }

  {
    auto mul = [] __device__(value_type lhs, value_type rhs, index_type a_col) -> value_type {
      value_type mult_res = max(lhs, rhs);
      mult_res >>= sizeof(value_type) * 8 / 2;
      mult_res++;
      mult_res <<= sizeof(value_type) * 8 / 2;
      mult_res += a_col;
      return mult_res;
    };

    auto add = [] __device__(value_type * lhs, value_type rhs) -> void {
      //        static_assert(sizeof(unsigned long long) == sizeof(value_type));
//              atomicMin((unsigned long long*)lhs, (unsigned long long)rhs);
      *lhs = min(*lhs, rhs);
    };

    nsparse::masked_spgemm_functor_t<value_type, index_type, zero, decltype(mul), decltype(add)>
        masked_spgemm(mul, add);

    for (auto& item : evaluation_plan) {
      masked_spgemm(masked_matrices[std::get<0>(item)], masked_matrices[std::get<1>(item)],
                    masked_matrices[std::get<2>(item)]);
    }
    cudaDeviceSynchronize();
  }
}

int nsparse_cfpq(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
                 const char** relations_names, size_t relations_count, size_t graph_size) {
  auto t1 = high_resolution_clock::now();

  size_t nonterm_count = grammar->nontermMapper.count;

  using index_type = unsigned int;

  std::vector<nsparse::matrix<bool, index_type>> matrices(
      nonterm_count, {static_cast<index_type>(graph_size), static_cast<index_type>(graph_size)});

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

          thrust::device_vector<index_type> col_index(col_idx, col_idx + nvals);
          thrust::device_vector<index_type> row_index(row_idx, row_idx + nrows + 1);

          matrices[simpleRule->l] = {
              std::move(col_index), std::move(row_index), static_cast<index_type>(graph_size),
              static_cast<index_type>(graph_size), static_cast<index_type>(nvals)};

          delete[] col_idx;
          delete[] row_idx;
          delete[](bool*) vals;

          GrB_Descriptor_free(&desc);
        }
      }
    }
  }

  std::vector<nsparse::matrix<bool, index_type>> matrices_copied(matrices);

  auto t2 = high_resolution_clock::now();
  response->time_to_prepare += duration<double, seconds::period>(t2 - t1).count();

  std::vector<std::tuple<int, int, int>> evaluation_plan;
  {
    std::vector<bool> changed(nonterm_count, true);
    std::vector<uint> sizes_before_before(nonterm_count, 0);

    nsparse::spgemm_functor_t<bool, index_type> spgemm_functor;

    bool matrices_is_changed = true;
    while (matrices_is_changed) {
      matrices_is_changed = false;
      response->iteration_count++;

      std::vector<uint> sizes_before(nonterm_count);
      for (auto i = 0; i < nonterm_count; i++) {
        sizes_before[i] = matrices[i].m_vals;
      }

      for (int i = 0; i < grammar->complex_rules_count; ++i) {
        MapperIndex nonterm1 = grammar->complex_rules[i].l;
        MapperIndex nonterm2 = grammar->complex_rules[i].r1;
        MapperIndex nonterm3 = grammar->complex_rules[i].r2;

        if (!changed[nonterm2] && !changed[nonterm3])
          continue;

        if (matrices[nonterm2].m_vals == 0 || matrices[nonterm3].m_vals == 0)
          continue;

        GrB_Index nvals_new, nvals_old;

        nvals_old = matrices[nonterm1].m_vals;

        auto new_mat = spgemm_functor(matrices[nonterm1], matrices[nonterm2], matrices[nonterm3]);
        nvals_new = new_mat.m_vals;

        if (nvals_new != nvals_old) {
          evaluation_plan.emplace_back(nonterm1, nonterm2, nonterm3);
          matrices_is_changed = true;
          changed[nonterm1] = true;
        }

        matrices[nonterm1] = std::move(new_mat);
      }

      for (auto i = 0; i < nonterm_count; i++) {
        if (sizes_before[i] == matrices[i].m_vals) {
          changed[i] = false;
        }
        sizes_before_before[i] = sizes_before[i];
      }
    }
  }

  using value_type = uint64_t;

  cudaDeviceSynchronize();
  t1 = high_resolution_clock::now();
  index_path<value_type, index_type>(matrices_copied, matrices, evaluation_plan, graph_size,
                                     nonterm_count);
  t2 = high_resolution_clock::now();
  response->time_to_index_path += duration<double, seconds::period>(t2 - t1).count();

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char* nonterm;

    nvals = matrices[i].m_vals;
    nonterm = ItemMapper_Map((ItemMapper*)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}
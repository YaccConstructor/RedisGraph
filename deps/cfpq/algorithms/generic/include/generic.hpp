#pragma once
#include <cassert>
#include <vector>
#include <chrono>
#include <grammar.h>
#include <item_mapper.h>
#include <algorithm>
#include <tuple>

namespace algorithms {

template <typename Mat>
std::vector<Mat> matrix_init(const Grammar* grammar, CfpqResponse* response,
                             const GrB_Matrix* relations, const char** relations_names,
                             size_t relations_count, size_t graph_size) {
  auto nonterm_count = grammar->nontermMapper.count;

  std::vector<GrB_Matrix> grb_matrices(nonterm_count);
  for (uint64_t i = 0; i < nonterm_count; i++) {
    GrB_Info info = GrB_Matrix_new(&grb_matrices[i], GrB_BOOL, graph_size, graph_size);
    assert(info == GrB_SUCCESS);
  }

  GrB_Monoid bool_monoid;
  GrB_Info info = GrB_Monoid_new_BOOL(&bool_monoid, GrB_LOR, false);
  assert(info == GrB_SUCCESS);

  for (int i = 0; i < relations_count; i++) {
    const char* terminal = relations_names[i];

    MapperIndex terminal_id =
        ItemMapper_GetPlaceIndex((ItemMapper*)&grammar->tokenMapper, terminal);
    if (terminal_id != grammar->tokenMapper.count) {
      for (int j = 0; j < grammar->simple_rules_count; j++) {
        const SimpleRule* simpleRule = &grammar->simple_rules[j];
        if (simpleRule->r == terminal_id) {
          GrB_Matrix result;
          info = GrB_Matrix_new(&result, GrB_BOOL, graph_size, graph_size);
          assert(info == GrB_SUCCESS);
          info = GrB_eWiseAdd_Matrix_Monoid(result, GrB_NULL, GrB_NULL, bool_monoid, relations[i],
                                            grb_matrices[simpleRule->l], GrB_NULL);
          assert(info == GrB_SUCCESS);
          GrB_Matrix_free(&grb_matrices[simpleRule->l]);
          grb_matrices[simpleRule->l] = result;
        }
      }
    }
  }

  GrB_Monoid_free(&bool_monoid);


  auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<Mat> matrices;
  for (auto i = 0; i < nonterm_count; i++) {
    matrices.emplace_back(Mat(grb_matrices[i]));
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  response->time_to_prepare +=
      std::chrono::duration<double, std::chrono::seconds::period>(t2 - t1).count();
  return matrices;
}

template <typename Mat, typename Func>
std::pair<int, std::vector<std::tuple<int, int, int>>> matrix_closure(const Grammar* grammar,
                                                                      std::vector<Mat>& matrices) {
  size_t nonterm_count = grammar->nontermMapper.count;

  assert(nonterm_count == matrices.size());

  std::vector<std::tuple<int, int, int>> evaluation_plan;
  std::vector<bool> changed(nonterm_count, true);

  Func spgemm_functor{};

  int iteration_count = 0;

  while (std::find(changed.begin(), changed.end(), true) != changed.end()) {
    iteration_count++;

    std::vector<uint> sizes_before(nonterm_count);
    for (auto i = 0; i < nonterm_count; i++) {
      sizes_before[i] = matrices[i].vals();
    }

    for (int i = 0; i < grammar->complex_rules_count; ++i) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      if (!changed[nonterm2] && !changed[nonterm3])
        continue;

      if (matrices[nonterm2].vals() == 0 || matrices[nonterm3].vals() == 0)
        continue;

      auto vals_before = matrices[nonterm1].vals();
      auto new_mat = spgemm_functor(matrices[nonterm1], matrices[nonterm2], matrices[nonterm3]);
      auto vals_after = new_mat.vals();

      if (vals_after != vals_before) {
        changed[nonterm1] = true;
        evaluation_plan.emplace_back(nonterm1, nonterm2, nonterm3);
      }

      matrices[nonterm1] = std::move(new_mat);
    }

    for (auto i = 0; i < nonterm_count; i++) {
      changed[i] = sizes_before[i] != matrices[i].vals();
    }
  }

  return {iteration_count, evaluation_plan};
}

template <typename Mat>
void fill_response(const Grammar* grammar, const std::vector<Mat>& matrices, CfpqResponse* response,
                   int iterations) {
  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    auto vals = matrices[i].vals();
    auto nonterm = ItemMapper_Map((ItemMapper*)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, vals);
  }
  response->iteration_count = iterations;
}

}  // namespace algorithms
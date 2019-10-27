#include "m4ri.h"
#include "methodOf4RusBooleanSemiringMatrix.h"
#include <algorithm>
#include <item_mapper.h>

int m4ri(const Grammar *grammar, CfpqResponse *response,
         const GrB_Matrix *relations, const char **relations_names,
         size_t relations_count, size_t graph_size) {
  // Create matrices
  uint64_t nonterm_count = grammar->nontermMapper.count;
  std::vector<Matrix *> matrices;
  for (size_t i = 0; i < nonterm_count; i++) {
    matrices.emplace_back(new MethodOf4RusMatrix(graph_size));
  }
  // Initialize matrices
  for (size_t i = 0; i < relations_count; i++) {
    const char *terminal = relations_names[i];

    MapperIndex terminal_id =
        ItemMapper_GetPlaceIndex((ItemMapper *)&grammar->tokenMapper, terminal);
    if (terminal_id != grammar->tokenMapper.count) {
      for (int j = 0; j < grammar->simple_rules_count; j++) {
        const SimpleRule *simpleRule = &grammar->simple_rules[j];
        if (simpleRule->r == terminal_id) {
          std::vector<GrB_Index> I, J;
          GrB_Index nvals;
          GrB_Matrix_nvals(&nvals, relations[i]);
          I.resize(nvals);
          J.resize(nvals);
          GrB_Matrix_extractTuples_BOOL(I.data(), J.data(), nullptr, &nvals,
                                        relations[i]);
          for (size_t i = 0; i < nvals; i++) {
            matrices[simpleRule->l]->set_bit(I[i], J[i]);
          }
        }
      }
    }
  }

  MethodOf4RusMatricesEnv env;
  env.environment_preprocessing(matrices);

  bool matrices_is_changed = true;
  while (matrices_is_changed) {
    matrices_is_changed = false;

    for (int i = 0; i < grammar->complex_rules_count; ++i) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      matrices_is_changed |=
          matrices[nonterm1]->add_mul(matrices[nonterm2], matrices[nonterm3]);
    }
  }

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char *nonterm;

    nvals = matrices[i]->bit_count();

    nonterm = ItemMapper_Map((ItemMapper *)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  env.environment_postprocessing(matrices);

  std::for_each(matrices.begin(), matrices.end(),
                [](Matrix *ptr) { delete ptr; });
  return 0;
}

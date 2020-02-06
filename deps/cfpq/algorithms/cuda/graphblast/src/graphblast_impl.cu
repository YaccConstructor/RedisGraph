#define GRB_USE_CUDA
#define private public

#include <grammar.h>
#include <item_mapper.h>
#include <map>
#include <response.h>
#include <set>

#include <algorithm>
#include <iostream>

#include "graphblas/graphblas.hpp"

int graphblast_impl(
    const Grammar *grammar, CfpqResponse *response,
    const std::map<MapperIndex, std::set<std::pair<GrB_Index, GrB_Index>>>
        &sparse_matrices,
    size_t graph_size) {

  std::map<MapperIndex, graphblas::Matrix<float>> matrices;

  {
    std::vector<graphblas::Index> rows;
    std::vector<graphblas::Index> cols;
    for (const auto &pair : sparse_matrices) {
      rows.clear();
      cols.clear();
      rows.reserve(pair.second.size());
      cols.reserve(pair.second.size());
      std::for_each(pair.second.begin(), pair.second.end(), [&](auto &pair) {
        rows.push_back(pair.first);
        cols.push_back(pair.second);
      });

      std::vector<float> value(pair.second.size(), 1);
      matrices[pair.first] = graphblas::Matrix<float>(graph_size, graph_size);
      matrices[pair.first].build(&rows, &cols, &value, pair.second.size(),
                                 GrB_NULL);
    }
  }

  graphblas::LogicalOrAndSemiring<float> semiring;

  bool matrices_is_changed = true;
  while (matrices_is_changed) {
    matrices_is_changed = false;
    response->iteration_count++;
    for (int i = 0; i < grammar->complex_rules_count; i++) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      graphblas::Index nvals_old;
      matrices[nonterm1].nvals(&nvals_old);
      graphblas::Descriptor desc;

      graphblas::mxm<float, float, float, float>(
          &matrices[nonterm1], GrB_NULL, logical_or<>{}, semiring,
          &matrices[nonterm2], &matrices[nonterm3], &desc);

      graphblas::Index nvals_new;
      matrices[nonterm1].nvals(&nvals_new);
      if (nvals_new != nvals_old) {
        matrices_is_changed = true;
      }
    }
  }

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    graphblas::Index nvals;
    char *nonterm;

    matrices[i].nvals(&nvals);
    nonterm = ItemMapper_Map((ItemMapper *)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}

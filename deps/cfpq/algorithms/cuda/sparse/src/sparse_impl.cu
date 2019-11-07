#include "sparse_impl.h"
#include <thrust/detail/config/compiler_fence.h>
#include <cusp/coo_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/multiply.h>
#include <item_mapper.h>

int sparse_impl(
    const Grammar *grammar, CfpqResponse *response,
    const std::map<MapperIndex, std::set<std::pair<GrB_Index, GrB_Index>>>
        &sparse_matrices,
    size_t graph_size) {

  std::vector<cusp::coo_matrix<GrB_Index, bool, cusp::device_memory>> matrices(
      grammar->nontermMapper.count, {graph_size, graph_size, 0});

  for (const auto &elem : sparse_matrices) {
    const auto &index = elem.first;
    const auto &values = elem.second;

    using IndexArray = cusp::array1d<GrB_Index, cusp::host_memory>;
    using ValueArray = cusp::array1d<bool, cusp::host_memory>;

    ValueArray value(values.size(), true);
    IndexArray rows, cols;

    rows.reserve(values.size());
    cols.reserve(values.size());
    std::for_each(values.begin(), values.end(), [&](auto &pair) {
      rows.push_back(pair.first);
      cols.push_back(pair.second);
    });

    cusp::coo_matrix_view<IndexArray::view, IndexArray::view, ValueArray::view>
        view(graph_size, graph_size, values.size(), rows, cols, value);

    matrices[index] =
        cusp::coo_matrix<GrB_Index, bool, cusp::device_memory>(view);
  }

  bool matrices_is_changed = true;
  while (matrices_is_changed) {
    matrices_is_changed = false;

    for (int i = 0; i < grammar->complex_rules_count; ++i) {
      MapperIndex nonterm1 = grammar->complex_rules[i].l;
      MapperIndex nonterm2 = grammar->complex_rules[i].r1;
      MapperIndex nonterm3 = grammar->complex_rules[i].r2;

      auto before = matrices[nonterm1].num_entries;

      cusp::coo_matrix<GrB_Index, bool, cusp::device_memory> C, D;
      cusp::multiply(matrices[nonterm2], matrices[nonterm3], C);
      cusp::add(C, matrices[nonterm1], D);
      matrices[nonterm1] = std::move(D)

      auto after = matrices[nonterm1].num_entries;

      matrices_is_changed |= (before != after);
    }
  }

  for (int i = 0; i < grammar->nontermMapper.count; i++) {
    size_t nvals;
    char *nonterm;

    nvals = matrices[i].num_entries;
    nonterm = ItemMapper_Map((ItemMapper *)&grammar->nontermMapper, i);
    CfpqResponse_Append(response, nonterm, nvals);
  }

  return 0;
}

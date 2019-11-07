#include "sparse.h"
#include "sparse_impl.h"

#include <GraphBLAS.h>
#include <cusp/coo_matrix.h>
#include <cusparse.h>
#include <item_mapper.h>
#include <map>
#include <set>

int sparse(const Grammar *grammar, CfpqResponse *response,
           const GrB_Matrix *relations, const char **relations_names,
           size_t relations_count, size_t graph_size) {

  std::map<MapperIndex, std::set<std::pair<GrB_Index, GrB_Index>>>
      sparse_matrices;

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
            sparse_matrices[simpleRule->l].insert(std::make_pair(I[i], J[i]));
          }
        }
      }
    }
  }

  sparse_impl(grammar, response, sparse_matrices, graph_size);

  return 0;
}

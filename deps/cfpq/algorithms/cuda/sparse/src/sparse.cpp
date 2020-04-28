#include "sparse.h"

#include <GraphBLAS.h>
#include <cusp/coo_matrix.h>
#include <cusparse.h>
#include <item_mapper.h>
#include <map>
#include <set>
#include <chrono>

using namespace std::chrono;

int sparse_impl(
    const Grammar* grammar, CfpqResponse* response,
                std::map<MapperIndex, std::pair<std::vector<GrB_Index>, std::vector<GrB_Index>>>
                    sparse_matrices,
    size_t graph_size);

int sparse(const Grammar* grammar, CfpqResponse* response, const GrB_Matrix* relations,
           const char** relations_names, size_t relations_count, size_t graph_size) {
  auto t1 = high_resolution_clock::now();

  std::map<MapperIndex, std::pair<std::vector<GrB_Index>, std::vector<GrB_Index>>> sparse_matrices;

  for (size_t i = 0; i < relations_count; i++) {
    const char* terminal = relations_names[i];

    MapperIndex terminal_id =
        ItemMapper_GetPlaceIndex((ItemMapper*)&grammar->tokenMapper, terminal);
    if (terminal_id != grammar->tokenMapper.count) {
      for (int j = 0; j < grammar->simple_rules_count; j++) {
        const SimpleRule* simpleRule = &grammar->simple_rules[j];
        if (simpleRule->r == terminal_id) {
          GrB_Index nvals;
          GrB_Matrix_nvals(&nvals, relations[i]);
          std::vector<GrB_Index> I, J;
          I.resize(nvals);
          J.resize(nvals);
          GrB_Matrix_extractTuples_BOOL(I.data(), J.data(), nullptr, &nvals, relations[i]);

          sparse_matrices[simpleRule->l] = std::make_pair(std::move(I), std::move(J));
        }
      }
    }
  }

  auto t2 = high_resolution_clock::now();
  response->time_to_prepare += duration<double, seconds::period>(t2 - t1).count();

  sparse_impl(grammar, response, std::move(sparse_matrices), graph_size);

  return 0;
}

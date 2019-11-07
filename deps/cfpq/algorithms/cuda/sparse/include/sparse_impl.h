#pragma once
#include "sparse.h"
#include <map>
#include <set>

int sparse_impl(
    const Grammar *grammar, CfpqResponse *response,
    const std::map<MapperIndex, std::set<std::pair<GrB_Index, GrB_Index>>>
        &sparse_matrices,
    size_t graph_size);

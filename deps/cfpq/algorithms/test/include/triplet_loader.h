#pragma once

#include <map>

extern "C" {
#include <GraphBLAS.h>
}

std::map<std::string, GrB_Matrix> load_triplets(std::istream &istream);

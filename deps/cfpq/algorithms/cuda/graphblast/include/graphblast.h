#pragma once

#include <grammar.h>
#include <response.h>

#ifdef __cplusplus
extern "C" {
#endif

int graphblast(const Grammar *grammar, CfpqResponse *response,
           const GrB_Matrix *relations, const char **relations_names,
           size_t relations_count, size_t graph_size);

#ifdef __cplusplus
}
#endif

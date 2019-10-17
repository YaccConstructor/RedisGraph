#pragma once

#include <conf.h>
#include <GraphBLAS.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    MapperIndex count;
    char nonterms[MAX_NONTERM_COUNT][MAX_ITEM_NAME_LEN];
    GrB_Index control_sums[MAX_NONTERM_COUNT];
} CfpqResponse;

void CfpqResponse_Init(CfpqResponse *resp);
int CfpqResponse_Append(CfpqResponse *resp, const char* nonterm, GrB_Index control_sum);

#ifdef __cplusplus
}
#endif

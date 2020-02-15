#pragma once

#include "../grammar/conf.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"
#include "../redismodule.h"

struct CustomResponseBase {
    void (*reply) (struct CustomResponseBase *, RedisModuleCtx *);
    void (*free) (struct CustomResponseBase *);
};
typedef struct CustomResponseBase CustomResponseBase;

typedef struct {
    uint64_t iteration_count;
    int rss_dif;
    int vms_dif;
    int shared_dif;

    MapperIndex count;
    char nonterms[MAX_NONTERM_COUNT][MAX_ITEM_NAME_LEN];
    GrB_Index control_sums[MAX_NONTERM_COUNT];

    CustomResponseBase *customResp;
} CfpqResponse;


void CfpqResponse_Init(CfpqResponse *resp);
int CfpqResponse_Append(CfpqResponse *resp, const char* nonterm, GrB_Index control_sum);

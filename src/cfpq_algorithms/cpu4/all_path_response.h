#pragma once

#include "../response.h"

typedef struct {
    GrB_Index left;
    GrB_Index right;
    uint32_t length;
    double time;
} PathResponse;

typedef struct {
    CustomResponseBase base;
    double index_time;
//    PathResponse *arr;
} AllPathResponse;

AllPathResponse* AllPathResponse_New(size_t cap);
void all_path_reply(CustomResponseBase *base, RedisModuleCtx *ctx);

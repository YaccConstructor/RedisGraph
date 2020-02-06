#pragma once

#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include <grammar/include/grammar.h>
#include <response/include/response.h>

int CFPQ_cpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
int CFPQ_gpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
int CFPQ_gpu2(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
//int CFPQ_gpu3(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
#pragma once

#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../grammar/grammar.h"
#include "response.h"

int CFPQ_cpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
int CFPQ_cpu3(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
int CFPQ_cpu4(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
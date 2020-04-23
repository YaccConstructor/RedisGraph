#pragma once

#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../grammar/grammar.h"
#include "../automat/automat.h"
#include "../bool_automat/bool_automat.h"
#include "response.h"

int CFPQ_cpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response);
//int CFPQ_tensor(RedisModuleCtx *ctx, GraphContext *gc, automat *grammar, CfpqResponse *response);
int CFPQ_tensor_new(RedisModuleCtx *ctx, GraphContext *gc, automat *grammar, CfpqResponse *response);
int CFPQ_cpu2(RedisModuleCtx *ctx, GraphContext* gc, bool_automat *grammar, CfpqResponse* response);

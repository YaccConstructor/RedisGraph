#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../bool_automat/bool_automat.h"
#include "../automat/automat.h"
#include "../cfpq_algorithms/algo_registrator.h"
#include "../cfpq_algorithms/response.h"
#include "../util/simple_timer.h"

int MGraph_CFPQ(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    printf("start");
    if (argc != 4) {
        RedisModule_ReplyWithError(ctx, "expected 3 args: algorithm name, graph name, grammar file path");
    }

    const char* algo_name = RedisModule_StringPtrLen(argv[1], NULL);;
    const char* graph_name = RedisModule_StringPtrLen(argv[2], NULL);
    const char* grammar_file_path = RedisModule_StringPtrLen(argv[3], NULL);

    // Load graph
    char msg[100];
    GraphContext *gc = GraphContext_Retrieve(ctx, graph_name, true);
    if (gc == NULL) {
        sprintf(msg, "): Graph \"%s\" not found :(", graph_name);
        RedisModule_ReplyWithError(ctx, msg);
        return REDISMODULE_ERR;
    }

    // Load grammar
    FILE* f = fopen(grammar_file_path, "r");
    bool_automat grammar;
    if (f == NULL)
    {
	sprintf(msg, "): File \"%s\" not found :(", grammar_file_path);
        RedisModule_ReplyWithError(ctx, msg);
        return REDISMODULE_ERR;
    }
    automat_bool_load(&grammar, f);
    fclose(f);

    // Check algo exist
    AlgoPointer_bool_automat algo = AlgoStorage_bool_automat_Get(algo_name);
    if (algo == NULL) {
        sprintf(msg, "): Algorithm \"%s\" not registered :(", algo_name);
        RedisModule_ReplyWithError(ctx, msg);
        return REDISMODULE_ERR;
    }

    // Start algorithm
    double timer[2];

    CfpqResponse response;
    CfpqResponse_Init(&response);

    simple_tic(timer);
    algo(ctx, gc, &grammar, &response);
    double time_spent = simple_toc(timer);

    // Reply
    char raw_response[MAX_ITEM_NAME_LEN + 40];
    RedisModule_ReplyWithArray(ctx, 3 + (response.customResp != NULL ? 1 : 0));

    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithSimpleString(ctx, "time");
    RedisModule_ReplyWithDouble(ctx, time_spent);

    printf("%d" , response.rss_dif);
    printf("%d" , response.vms_dif);
    //RedisModule_ReplyWithArray(ctx, 2);
    //RedisModule_ReplyWithSimpleString(ctx, "mem rss");
    //RedisModule_ReplyWithLongLong(ctx, response.rss_dif);
  
    //RedisModule_ReplyWithArray(ctx, 2);
    //RedisModule_ReplyWithSimpleString(ctx, "mem vms");
    //RedisModule_ReplyWithLongLong(ctx, response.vms_dif);


    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithSimpleString(ctx, "iters");
    RedisModule_ReplyWithLongLong(ctx, response.iteration_count);

    RedisModule_ReplyWithArray(ctx, response.count);
    for (int i = 0; i < response.count; ++i) {
        RedisModule_ReplyWithArray(ctx, 2);
        RedisModule_ReplyWithSimpleString(ctx, response.nonterms[i]);
        RedisModule_ReplyWithLongLong(ctx, response.control_sums[i]);
    }

    if (response.customResp != NULL) {
        response.customResp->reply(response.customResp, ctx);
        response.customResp->free(response.customResp);
    }


    return REDISMODULE_OK;
}
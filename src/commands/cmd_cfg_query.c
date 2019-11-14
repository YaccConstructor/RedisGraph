#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../grammar/grammar.h"
#include "../cfpq_algorithms/algo_registrator.h"
#include "../cfpq_algorithms/response.h"
#include "../util/simple_timer.h"

int MGraph_CFPQ(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
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
    Grammar grammar;
    if (f == NULL) {
        sprintf(msg, "): File \"%s\" not found :(", grammar_file_path);
        RedisModule_ReplyWithError(ctx, msg);
        return REDISMODULE_ERR;
    }
    if (Grammar_Load(&grammar, f) != GRAMMAR_LOAD_SUCCESS) {
        RedisModule_ReplyWithError(ctx, "): Grammar has not loaded :(");
        return REDISMODULE_ERR;
    }
    fclose(f);

    // Check algo exist
    AlgoPointer algo = AlgoStorage_Get(algo_name);
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
    RedisModule_ReplyWithArray(ctx, response.count + 5);

    sprintf(raw_response, "Time spent: %f", time_spent);
    RedisModule_ReplyWithSimpleString(ctx, raw_response);

    sprintf(raw_response, "VMS delta: %d", response.vms_dif);
    RedisModule_ReplyWithSimpleString(ctx, raw_response);

    sprintf(raw_response, "RSS delta: %d", response.rss_dif);
    RedisModule_ReplyWithSimpleString(ctx, raw_response);

    sprintf(raw_response, "Shared delta: %d", response.shared_dif);
    RedisModule_ReplyWithSimpleString(ctx, raw_response);

    sprintf(raw_response, "Iteration count: %lu", response.iteration_count);
    RedisModule_ReplyWithSimpleString(ctx, raw_response);

    for (int i = 0; i < response.count; ++i) {
        sprintf(raw_response, "%s: %lu", response.nonterms[i], response.control_sums[i]);
        RedisModule_ReplyWithSimpleString(ctx, raw_response);
    }
    return REDISMODULE_OK;
}
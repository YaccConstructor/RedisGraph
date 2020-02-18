#include "all_path_response.h"
#include "../../util/arr.h"

void all_path_reply(CustomResponseBase *base, RedisModuleCtx *ctx) {
    /*
     *1) 1) index time
     *   2) 1) (i, j, length, time)
     *      2) ...
     */

    AllPathResponse *resp = (AllPathResponse *) base;

//    RedisModule_ReplyWithArray(ctx, 2);

    RedisModule_ReplyWithDouble(ctx, resp->index_time);

//    RedisModule_ReplyWithArray(ctx, array_len(resp->arr));
//    for (int i = 0; i < array_len(resp->arr); ++i) {
//        RedisModule_ReplyWithArray(ctx, 4);
//        RedisModule_ReplyWithLongLong(ctx, resp->arr[i].left);
//        RedisModule_ReplyWithLongLong(ctx, resp->arr[i].right);
//        RedisModule_ReplyWithLongLong(ctx, resp->arr[i].length);
//        RedisModule_ReplyWithDouble(ctx, resp->arr[i].time);
//    }
}

void all_path_free(CustomResponseBase *base) {
    AllPathResponse *allPathResponse = (AllPathResponse *) base;
//    array_free(allPathResponse->arr);
    free(allPathResponse);
}

AllPathResponse* AllPathResponse_New(size_t cap) {
    AllPathResponse *resp = malloc(sizeof(AllPathResponse));
    resp->index_time = 0;
//    resp->arr = array_new(PathResponse, cap);

    resp->base.reply = all_path_reply;
    resp->base.free = all_path_free;

    return resp;
}


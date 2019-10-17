#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "cfpq_algorithms.h"

#define MAX_ALGO_COUNT 100 // why not :D
#define MAX_ALGO_NAME 100

typedef int (*AlgoPointer)(RedisModuleCtx*, GraphContext*, Grammar*, CfpqResponse*);

typedef struct {
    int count;
    char names[MAX_ALGO_COUNT][MAX_ALGO_NAME];
    AlgoPointer algorithms[MAX_ALGO_COUNT];
} AlgoStorage;


void AlgoStorage_Init();
int AlgoStorage_Add(const char *name, AlgoPointer algo);
AlgoPointer AlgoStorage_Get(const char *name);
int AlgoStorage_Count();

void AlgoStorage_RegisterAlgorithms();
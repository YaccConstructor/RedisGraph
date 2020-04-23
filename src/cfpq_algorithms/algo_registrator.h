#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "cfpq_algorithms.h"
#include "response.h"

#define MAX_ALGO_COUNT 100 // why not :D
#define MAX_ALGO_NAME 100

typedef int (*AlgoPointer)(RedisModuleCtx*, GraphContext*, Grammar*, CfpqResponse*);
typedef int (*AlgoPointer_automat)(RedisModuleCtx*, GraphContext*, automat*, CfpqResponse*);
typedef int (*AlgoPointer_bool_automat)(RedisModuleCtx*, GraphContext*, bool_automat*, CfpqResponse*);


typedef struct {
    int count;
    char names[MAX_ALGO_COUNT][MAX_ALGO_NAME];
    AlgoPointer algorithms[MAX_ALGO_COUNT];
} AlgoStorage;

typedef struct {
    int count;
    char names[MAX_ALGO_COUNT][MAX_ALGO_NAME];
    AlgoPointer_automat algorithms[MAX_ALGO_COUNT];
} AlgoStorage_automat;

typedef struct {
    int count;
    char names[MAX_ALGO_COUNT][MAX_ALGO_NAME];
    AlgoPointer_bool_automat algorithms[MAX_ALGO_COUNT];
} AlgoStorage_bool_automat;


void AlgoStorage_Init();

void AlgoStorage_automat_Init();
void AlgoStorage_bool_automat_Init();

int AlgoStorage_Add(const char *name, AlgoPointer algo);

int AlgoStorage_automat_Add(const char *name, AlgoPointer_automat algo);
int AlgoStorage_bool_automat_Add(const char *name, AlgoPointer_bool_automat algo);

AlgoPointer AlgoStorage_Get(const char *name);

AlgoPointer_automat AlgoStorage_automat_Get(const char *name);
AlgoPointer_bool_automat AlgoStorage_bool_automat_Get(const char *name);

int AlgoStorage_Count();

int AlgoStorage_automat_Count();
int AlgoStorage_bool_automat_Count();

void AlgoStorage_RegisterAlgorithms();
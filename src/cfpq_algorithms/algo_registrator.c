#include "algo_registrator.h"

AlgoStorage CfpqAlgoStorage;

void AlgoStorage_Init() {
    CfpqAlgoStorage.count = 0;
}

int AlgoStorage_Add(const char *name, AlgoPointer algo) {
    assert(CfpqAlgoStorage.count != MAX_ALGO_COUNT);

    strcpy(CfpqAlgoStorage.names[CfpqAlgoStorage.count], name);
    CfpqAlgoStorage.algorithms[CfpqAlgoStorage.count] = algo;

    return CfpqAlgoStorage.count++;
}

int AlgoStorage_automat_Add(const char *name, AlgoPointer_automat algo) {
    assert(CfpqAlgoStorage.count != MAX_ALGO_COUNT);

    strcpy(CfpqAlgoStorage.names[CfpqAlgoStorage.count], name);
    CfpqAlgoStorage.algorithms[CfpqAlgoStorage.count] = algo;

    return CfpqAlgoStorage.count++;
}

AlgoPointer_automat AlgoStorage_Get(const char *name) {
    for (int i = 0; i < CfpqAlgoStorage.count; ++i) {
        if (strcmp(name, CfpqAlgoStorage.names[i]) == 0) {
            return CfpqAlgoStorage.algorithms[i];
        }
    }
    return NULL;
}

int AlgoStorage_Count() {
    return CfpqAlgoStorage.count;
}

inline void AlgoStorage_RegisterAlgorithms() {
    AlgoStorage_Init();
    AlgoStorage_Add("cpu", CFPQ_cpu1);
    AlgoStorage_automat_Add("tensor", CFPQ_tensor);
}
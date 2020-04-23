#include "algo_registrator.h"

AlgoStorage CfpqAlgoStorage;
AlgoStorage_bool_automat CfpqAlgoStorage_automat;
AlgoStorage_bool_automat CfpqAlgoStorage_bool_automat;

void AlgoStorage_Init() {
    CfpqAlgoStorage.count = 0;
}

void AlgoStorage_automat_Init() {
    CfpqAlgoStorage_automat.count = 0;
}

void AlgoStorage_bool_automat_Init() {
    CfpqAlgoStorage_bool_automat.count = 0;
}

int AlgoStorage_Add(const char *name, AlgoPointer algo) {
    assert(CfpqAlgoStorage.count != MAX_ALGO_COUNT);

    strcpy(CfpqAlgoStorage.names[CfpqAlgoStorage.count], name);
    CfpqAlgoStorage.algorithms[CfpqAlgoStorage.count] = algo;

    return CfpqAlgoStorage.count++;
}

int AlgoStorage_automat_Add(const char *name, AlgoPointer_automat algo) {
    assert(CfpqAlgoStorage_automat.count != MAX_ALGO_COUNT);

    strcpy(CfpqAlgoStorage_automat.names[CfpqAlgoStorage_automat.count], name);
    CfpqAlgoStorage_automat.algorithms[CfpqAlgoStorage_automat.count] = algo;

    return CfpqAlgoStorage_automat.count++;
}

int AlgoStorage_bool_automat_Add(const char *name, AlgoPointer_bool_automat algo) {
    assert(CfpqAlgoStorage_bool_automat.count != MAX_ALGO_COUNT);

    strcpy(CfpqAlgoStorage_bool_automat.names[CfpqAlgoStorage_bool_automat.count], name);
    CfpqAlgoStorage_bool_automat.algorithms[CfpqAlgoStorage_bool_automat.count] = algo;

    return CfpqAlgoStorage_bool_automat.count++;
}

AlgoPointer AlgoStorage_Get(const char *name) {
    for (int i = 0; i < CfpqAlgoStorage.count; ++i) {
        if (strcmp(name, CfpqAlgoStorage.names[i]) == 0) {
            return CfpqAlgoStorage.algorithms[i];
        }
    }
    return NULL;
}

AlgoPointer_automat AlgoStorage_automat_Get(const char *name) {
    for (int i = 0; i < CfpqAlgoStorage_automat.count; ++i) {
        if (strcmp(name, CfpqAlgoStorage_automat.names[i]) == 0) {
            return CfpqAlgoStorage_automat.algorithms[i];
        }
    }
    return NULL;
}

AlgoPointer_bool_automat AlgoStorage_bool_automat_Get(const char *name) {
    for (int i = 0; i < CfpqAlgoStorage_bool_automat.count; ++i) {
        if (strcmp(name, CfpqAlgoStorage_bool_automat.names[i]) == 0) {
            return CfpqAlgoStorage_bool_automat.algorithms[i];
        }
    }
    return NULL;
}

int AlgoStorage_Count() {
    return CfpqAlgoStorage.count;
}

int AlgoStorage_automat_Count() {
    return CfpqAlgoStorage_automat.count;
}

int AlgoStorage_bool_automat_Count() {
    return CfpqAlgoStorage_bool_automat.count;
}

inline void AlgoStorage_RegisterAlgorithms() {
    AlgoStorage_Init();
    AlgoStorage_Add("cpu", CFPQ_cpu1);
    //AlgoStorage_automat_Add("tensor", CFPQ_tensor);
    AlgoStorage_automat_Add("nt", CFPQ_tensor_new);
    AlgoStorage_bool_automat_Add("nt", CFPQ_cpu2);    
}



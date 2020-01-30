#pragma once

#include "../../deps/GraphBLAS/Include/GraphBLAS.h"
#include "../grammar/grammar.h"

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t middle;

    uint32_t height;
    uint32_t length;
} PathIndex;

extern PathIndex PathIndex_Identity;

void PathIndex_Init(PathIndex *index, uint32_t left, uint32_t right, uint32_t middle, uint32_t height, uint32_t length);
void PathIndex_InitIdentity(PathIndex *index);
void PathIndex_Copy(const PathIndex *from, PathIndex *to);

bool PathIndex_IsIdentity(const PathIndex *index);

void PathIndex_Mul(void *z, const void *x, const void *y);
void PathIndex_Add(void *z, const void *x, const void *y);

void PathIndex_ToStr(const PathIndex *index, char *buf);
void PathIndex_Show(PathIndex *index);


void PathIndex_MatrixInit(GrB_Matrix *m, const GrB_Matrix *bool_m);
void PathIndex_MatrixShow(const GrB_Matrix *matrix);
void PathIndex_MatricesGetPath(const GrB_Matrix *matrices, const Grammar *grammar, GrB_Index left, GrB_Index right, MapperIndex nonterm);

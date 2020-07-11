#pragma once

#include "../automat/map.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"
#include <stdio.h>

#define SIZE_MATRIX 20

typedef struct
{
    bool m[SIZE_MATRIX][SIZE_MATRIX];
} bool_matrix;


typedef struct
{
    int count_matrices;
    int count_S_term;
    int64_t id[50];
    int size_matrices;
    GrB_Matrix matrices[15];
    map indices;
} bool_automat;

void automat_bool_load(bool_automat *aut, FILE *file);

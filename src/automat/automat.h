#pragma once

#include "Gvector.h"
#include "map.h"

typedef struct
{
    vector edges_i;
    vector edges_j;
    vector edges_label;
} edges;

typedef struct
{
    int init;

    int indicesCount;
    int statesCount;
    int startSymbolsCount;

    vector matrix;
    vector states;
    vector startSymbols;

    edges edg;
    edges paths;

    map *indices;
} automat;

void automat_delete(automat *aut);
void automat_load_from_file(automat *aut, char *file_name);
void automat_init(automat *aut);

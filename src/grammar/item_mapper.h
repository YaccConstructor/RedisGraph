#pragma once

#include <stdint.h>
#include "conf.h"

typedef struct {
    MapperIndex count;
    char items[];
} ItemMapper;

typedef enum {
    ItemMapper_EXIST,
    ItemMapper_NOT_EXIST
} IM_FindRes;

void ItemMapper_Init(ItemMapper *dict);

MapperIndex ItemMapper_Insert(ItemMapper *dict, char token);
IM_FindRes ItemMapper_Find(ItemMapper *dict, char token);
char ItemMapper_Map(ItemMapper *dict, char mapperIdex);

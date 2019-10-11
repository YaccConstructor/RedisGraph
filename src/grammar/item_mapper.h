#pragma once

#include <stdint.h>
#include "conf.h"

#define ITEM_NOT_EXIST 0
#define ITEM_EXIST 1

typedef struct {
    MapperIndex count;
    char items[][MAX_ITEM_NAME_LEN];
} ItemMapper;

void ItemMapper_Init(ItemMapper *dict);

MapperIndex ItemMapper_GetPlaceIndex(ItemMapper *dict, const char *token);
MapperIndex ItemMapper_Insert(ItemMapper *dict, const char* token);
int ItemMapper_Find(ItemMapper *dict, const char* token);
char* ItemMapper_Map(ItemMapper *dict, MapperIndex mapperIdex);

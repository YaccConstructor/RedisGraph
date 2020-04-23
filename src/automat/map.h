#pragma once

#include <stdint.h>
#include <string.h>

typedef int64_t MapIndex;

typedef struct
{
    int init;
    uint16_t count;
    MapIndex id_items[150];
    int64_t max_id_items;
    char items[150][30];
} map;

void map_init(map *m);
int map_get_size(map *m);
char *map_get_first_by_second(map *m, MapIndex index);
MapIndex map_get_second_by_first(map *m, char *f);
void map_append(map *m, char *f, MapIndex sec);


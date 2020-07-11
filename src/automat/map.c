#include "map.h"
#include <assert.h>
#include <stdio.h>
#include <limits.h>

void map_init(map *m)
{
    m->count = 0;
    m->init = 1;
}

int map_get_size(map *m)
{
    assert(m->init == 1);
    return m->count;
}

char *map_get_first_by_second(map *m, MapIndex index)
{
    assert(m->init == 1);
    assert(m->max_id_items > index);

    if (m->count == 0)
        return NULL;

    for (uint16_t i = 0; i < m->count; i++)
    {
        if (m->id_items[i] == index)
            return m->items[i];
    }

    return NULL;
}

MapIndex map_get_second_by_first(map *m, char *f)
{
    assert(m->init == 1);

    if (m->count == 0)
        return INT_MIN;
    //printf("%c", '\n');
    //printf("%s%s %c",  "\nf: ", f, '\t');
    for (uint16_t i = 0; i < m->count; i++)
    {
        //printf("%s", m->items[i]);
        //printf(" %d %c", strcmp(m->items[i], f), '\t');
        if (strcmp(m->items[i], f) == 0)
        {
           // printf("%d", 1);
            return m->id_items[i];
        }
    }
    //printf("%c", '\n');

    return INT_MIN;
}

void map_append(map *m, char *f, MapIndex sec)
{
    assert(m->init == 1);
    MapIndex index = map_get_second_by_first(m, f);
    if(index == INT_MIN)
    {
        strcpy(m->items[m->count], f);

        m->id_items[m->count] = sec;
        if (m->max_id_items < sec)
            m->max_id_items = sec;

        m->count++;
    }

}


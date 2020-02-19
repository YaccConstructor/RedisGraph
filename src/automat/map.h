#pragma once
#include <string.h>

typedef struct list_element
{
    char *first;
    int second;
    struct list_element *next;
    struct list_element *prev;
} list_map_element;

typedef struct
{
    list_map_element *head;
    list_map_element *tail;
    unsigned int size;
} list_map;

typedef struct
{
    list_map *m;
    int init;
} map;

void map_delete(map *m);
int map_get_size(map *m);
char *map_get_first_by_second(map *m, int sec); // считается, что между множеством char *first
int map_get_second_by_first(map *m, char *f);  //      и множеством int second имеется биекция
void map_init(map *m);
void map_append(map *m, char *f, int sec);

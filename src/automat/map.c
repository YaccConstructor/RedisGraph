#include "map.h"
#include <assert.h>
#include <limits.h>

void map_init(map *m)
{
    list_map *newm = (list_map *)malloc(sizeof (list_map));
    newm->head = NULL;
    newm->size = 0;
    m->m = newm;
    m->init = 1;
}

void list_map_delete(list_map *l)
{
    unsigned int size = l->size;
    list_map_element *current = l->head;
    list_map_element *next;

    while(size--)
    {
        next = current->next;
        free(current);
        current = next;
    }

    free(l);
}

void map_delete(map *m)
{
    assert(m->init == 1);
    list_map_delete(m->m);
    m->init = 0;
}

int map_get_size(map *m)
{
    assert(m->init == 1);
    return m->m->size;
}

char *map_get_first_by_second(map *m, int sec)
{
    assert(m->init == 1);

    if (m->m->size == 0)
        return NULL;

    list_map_element *current = m->m->head;
    while (current != NULL)
    {
        if (current->second == sec)
            return current->first;

        current = current->next;
    }

    return NULL;
}

int map_get_second_by_first(map *m, char *f)
{
    assert(m->init == 1);

    if (m->m->size == 0)
        return INT_MIN;

    list_map_element *current = m->m->head;
    while (current != NULL)
    {
        if (!strcmp(current->first, f))
            return current->second;

        current = current->next;
    }

    return INT_MIN;
}

void map_append(map *m, char *f, int sec)
{
    assert(m->init == 1);

    char *newf = malloc(sizeof (f));
    int i = 0;
    while (f[i] != '\0')
    {
        newf[i] = f[i];
        i++;
    }

    list_map_element *newnode = (list_map_element *)malloc(sizeof (list_map_element));
    newnode->first = newf;
    newnode->second = sec;

    if (m->m->size)
    {
        newnode->prev = m->m->tail;
        newnode->next = NULL;
        m->m->tail->next = newnode;
        m->m->tail = newnode;
    }
    else
    {
        newnode->next = NULL;
        newnode->prev = NULL;
        m->m->head = newnode;
        m->m->tail = newnode;
    }

    m->m->size++;
}

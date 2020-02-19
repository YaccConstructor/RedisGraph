#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include "automat.h"

void automat_init(automat *aut)
{
    vector_int_init(&aut->matrix);
    vector_int_init(&aut->states);
    vector_int_init(&aut->startSymbols);

    vector_int_init(&aut->edg.edges_i);
    vector_int_init(&aut->edg.edges_j);
    vector_int_init(&aut->edg.edges_label);

    vector_int_init(&aut->paths.edges_i);
    vector_int_init(&aut->paths.edges_j);
    vector_int_init(&aut->paths.edges_label);

    map_init(aut->indices);

    aut->init = 1;

    aut->statesCount = 0;
    aut->indicesCount = 0;
    aut->startSymbolsCount = 0;
}

void automat_delete(automat *aut)
{
    vector_int_delete(&aut->matrix);
    vector_int_delete(&aut->states);
    vector_int_delete(&aut->startSymbols);

    vector_int_delete(&aut->edg.edges_i);
    vector_int_delete(&aut->edg.edges_j);
    vector_int_delete(&aut->edg.edges_label);

    vector_int_delete(&aut->paths.edges_i);
    vector_int_delete(&aut->paths.edges_j);
    vector_int_delete(&aut->paths.edges_label);

    map_delete(aut->indices);

    aut->init = 0;
}

void automat_load_from_file(automat *aut, char *file_name)
{
    assert(aut->init == 1);

    FILE *file = fopen("input.txt", "r");
    assert(file != NULL);

    int edgesCount = 0;
    fscanf(file, "%d", &edgesCount);

    char skip;
    fscanf(file, "%c", &skip);

    char label[100];
    for (int i = 0; i < edgesCount; i++)
    {
        int k = 0;
        int l = 0;

        int count = fscanf(file, "%d%c%s%c%d%c", &k, &skip, label, &skip, &l, &skip);
//        fscanf(file, "%d", &k);
//        fscanf(file, "%c", skip);

//        fscanf(file, "%с", label);
//        fscanf(file, "%c", skip);

//        fscanf(file, "%d", &l);
//        fscanf(file, "%c", skip);

        int j = 0;
        int found = map_get_second_by_first(aut->indices, label);
        if (found == INT_MIN)
        {
             j = 1 << aut->indicesCount;
             aut->indicesCount++;
             map_append(aut->indices, label, j);
        }
        else
        {
            j = found;
        }

        vector_int_append(&aut->edg.edges_i, k);
        vector_int_append(&aut->edg.edges_j, l);
        vector_int_append(&aut->edg.edges_label, j);

        aut->statesCount = (k > aut->statesCount ? k: aut->statesCount);
        aut->statesCount = (l > aut->statesCount ? l: aut->statesCount);
    }

    aut->statesCount++;

    int pathsCount = 0;
    fscanf(file, "%d", &pathsCount);

    for (int i = 0; i < pathsCount; i++)
    {
        int k = 0;
        int l = 0;
        fscanf(file, "%d", &k);
        fscanf(file, "%s", label);
        fscanf(file, "%d", &l);

        int j = 0;
        int found = map_get_second_by_first(aut->indices, label);
        if (found == INT_MIN)
        {
             printf("%s%s", "Automata has incomplete type [name: ", file_name);
             continue;
        }
        else
        {
            j = found;
        }

        vector_int_append(&aut->paths.edges_i, k);
        vector_int_append(&aut->paths.edges_j, l);
        vector_int_append(&aut->paths.edges_label, j);
    }

    fscanf(file, "%d", &aut->startSymbolsCount);
    for (int i = 0; i < aut->startSymbolsCount; i++)
    {
        fscanf(file, "%s", label);

        int j = 0;
        int found = map_get_second_by_first(aut->indices, label);
        if (found == INT_MIN)
        {
             printf("%s%s", "Automata has incomplete type [name: ", file_name);
             continue;
        }
        else
        {
            j = found;
        }
        vector_int_append(&aut->startSymbols, label);
    }

    fclose(file);

    for (int i = 0; i < aut->statesCount * aut->statesCount; i++)
    {
        vector_int_append(&aut->matrix, 0);
        vector_int_append(&aut->states, 0);
    }

    for (int i = 0; i < vector_int_get_size(&aut->edg.edges_i); i++)
    {
        int e_i = vector_int_get_element_by_index(&aut->edg.edges_i, i);
        int e_j = vector_int_get_element_by_index(&aut->edg.edges_j, i);
        int e_label = vector_int_get_element_by_index(&aut->edg.edges_label, i);

        int newdata = vector_int_get_element_by_index(&aut->matrix, e_i*aut->statesCount + e_j);
        newdata |= e_label;

        vector_int_set_element(&aut->matrix, newdata, e_i*aut->statesCount + e_j);
    }

    for (int i = 0; i < vector_int_get_size(&aut->paths.edges_i); i++)
    {
        int p_i = vector_int_get_element_by_index(&aut->paths.edges_i, i);
        int p_j = vector_int_get_element_by_index(&aut->paths.edges_j, i);
        int p_label = vector_int_get_element_by_index(&aut->paths.edges_label, i);

        int newdata = vector_int_get_element_by_index(&aut->states, p_i*aut->statesCount + p_j);
        newdata |= p_label;

        vector_int_set_element(&aut->states, newdata, p_i*aut->statesCount + p_j);
    }




    printf("%d", aut->statesCount);
    printf("%c", '\n');
    printf("%c", '\n');

    for (int i = 0; i < aut->statesCount; i++)
    {
        for (int j = 0; j < aut->statesCount; j++)
            printf("%d", vector_int_get_element_by_index(&aut->matrix, i*aut->statesCount + j));

        printf("%c", '\n');
    }

    printf("%c", '\n');
    printf("%c", '\n');

    for (int i = 0; i < aut->statesCount; i++)
    {
        for (int j = 0; j < aut->statesCount; j++)
            printf("%d", vector_int_get_element_by_index(&aut->states, i*aut->statesCount + j));

        printf("%c", '\n');
    }

    printf("%c", '\n');
    printf("%c", '\n');

    for (int i = 0; i < vector_int_get_size(&aut->edg.edges_i); i++)
    {
        int e_i = vector_int_get_element_by_index(&aut->edg.edges_i, i);
        int e_j = vector_int_get_element_by_index(&aut->edg.edges_j, i);
        int e_label = vector_int_get_element_by_index(&aut->edg.edges_label, i);

        printf("%d %d %d", e_i, e_label, e_j);
        printf("%c", ' ');
        printf("%c", ' ');
        printf("%c", ' ');
    }

    printf("%c", '\n');
    printf("%c", '\n');

    for (int i = 0; i < vector_int_get_size(&aut->paths.edges_i); i++)
    {
        int p_i = vector_int_get_element_by_index(&aut->paths.edges_i, i);
        int p_j = vector_int_get_element_by_index(&aut->paths.edges_j, i);
        int p_label = vector_int_get_element_by_index(&aut->paths.edges_label, i);

        printf("%d %d %d", p_i, p_label, p_j);
        printf("%c", '\n');
    }
}

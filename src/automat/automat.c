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
    aut->init = 0;
}

void automat_load_from_file(automat *aut, FILE *file)
{
    automat_init(aut);
    char *buf;
    size_t buf_size = 0;

    getline(&buf, &buf_size, file);

    int edgesCount = 0;
    sscanf(buf, "%d", &edgesCount);

    char label[100];
    for (int i = 0; i < edgesCount; i++)
    {
        getline(&buf, &buf_size, file);
        int k = 0;
        int l = 0;

        sscanf(buf, "%d %100s %d", &k, label, &l);

        int j = 0;
        int found = map_get_second_by_first(&aut->indices, label);
        if (found == INT_MIN)
        {
             j = 1 << aut->indicesCount;
             aut->indicesCount++;
             map_append(&aut->indices, label, j);
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

    getline(&buf, &buf_size, file);
    int pathsCount = 0;
    sscanf(buf, "%d", &pathsCount);

    for (int i = 0; i < pathsCount; i++)
    {
        int k = 0;
        int l = 0;
        getline(&buf, &buf_size, file);
        sscanf(buf, "%d %100s %d", &k, label, &l);

        int j = 0;
        int found = map_get_second_by_first(&aut->indices, label);
        if (found == INT_MIN)
        {
	    printf("%s", "Automata has incomplete type");
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

    getline(&buf, &buf_size, file);
    sscanf(buf, "%d", &aut->startSymbolsCount);

    for (int i = 0; i < aut->startSymbolsCount; i++)
    {
        getline(&buf, &buf_size, file);
        sscanf(buf, "%100s", label);

        int j = 0;
        int found = map_get_second_by_first(&aut->indices, label);
        if (found == INT_MIN)
        {
             j = 1 << aut->indicesCount;
             aut->indicesCount++;
             map_append(&aut->indices, label, j); 
        }
        else
        {
	    j = found;
        }
        vector_int_append(&aut->startSymbols, j);
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


}

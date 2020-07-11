#include "bool_automat.h"
#include <assert.h>

//#include "../util/arr.h"

void automat_bool_init(bool_automat *aut, int count_matrices, int count_S_term, int size)
{
    assert(count_S_term + count_S_term < 15 && "Oy oy malo matrix in automat :)");

    map_init(&aut->indices);
    aut->count_matrices = count_matrices;
    aut->count_S_term = count_S_term;
    aut->size_matrices = size;
}

void automat_bool_load(bool_automat *aut, FILE *file)
{
    assert(file != NULL);

    int count_matrices = 0;
    fscanf(file, "%d", &count_matrices);
    int count_S_term = 0;
    fscanf(file, "%d", &count_S_term);
    int size = 0;
    fscanf(file, "%d", &size);

    automat_bool_init(aut, count_matrices, count_S_term, size);

	for (int i = 0; i < count_S_term + count_matrices; i++)
    {
        GrB_Info info = GrB_Matrix_new(&aut->matrices[i], GrB_BOOL, size, size);
        assert(info == GrB_SUCCESS);
    }

    for (int i = 0; i < count_S_term + count_matrices; i++)
    {
        GrB_Info info = GrB_Matrix_new(&aut->matrices[i], GrB_BOOL, size, size);
        assert(info == GrB_SUCCESS);
    }

    for (int i = 0; i < count_matrices; i++)
    {
        char label[10];
        fscanf(file, "%10s", label);
        int64_t id = 1 << i;
        map_append(&aut->indices, label, id);
        aut->id[i] = id;
        int count_edges = 0;
        fscanf(file, "%d", &count_edges);
        for (int j = 0; j < count_edges; j++)
        {
            int edges_i = 0;
            int edges_j = 0;
            fscanf(file, "%d %d", &edges_i, &edges_j);
            GrB_Info inf = GrB_Matrix_setElement_BOOL(aut->matrices[i], true, edges_i, edges_j);
            assert(inf == GrB_SUCCESS);
        }
    }

    for (int i = 0; i < count_S_term; i++)
    {
        char label[10];
        fscanf(file, "%10s", label);
        int id = map_get_second_by_first(&aut->indices, label);
	if (id == INT_MIN)
	{
	    id = 1 << (count_matrices + i);
	    map_append(&aut->indices, label, id);
	}
        aut->id[count_matrices + i] = id;

        int count_edges = 0;
        fscanf(file, "%d", &count_edges);
        for (int j = 0; j < count_edges; j++)
        {
            int edges_i = 0;
            int edges_j = 0;
            fscanf(file, "%d %d", &edges_i, &edges_j);
            GrB_Matrix_setElement_BOOL(aut->matrices[count_matrices + i], true, edges_i, edges_j);
        }
    }

    aut->id[count_S_term + count_matrices] = -1;
}

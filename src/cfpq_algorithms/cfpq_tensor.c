#include "cfpq_algorithms.h"
#include "../redismodule.h"
#include "../automat/automat.h"

#define F_BINARY(f) ((void (*)(void *, const void *, const void *)) f)

void bin_for_kron(int64_t *z, const int64_t *x, const int64_t *y)
{
    *z = *x & *y;

    if (*z != 0)
        *z = 1;
}

int CFPQ_tensor(RedisModuleCtx *ctx, GraphContext *gc, automat *grammar, CfpqResponse *response)
{
    // create automat matrix
    GrB_Matrix Automat;
    uint64_t sizeAutomat = grammar->statesCount;
    GrB_Info info = GrB_Matrix_new(&Automat, GrB_INT64, sizeAutomat, sizeAutomat);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");

    for (int i = 0; i < sizeAutomat; i++)
    {
        for (int j = 0; j < sizeAutomat; j++)
            GrB_Matrix_setElement_INT64(Automat, vector_int_get_element_by_index(&grammar->matrix, i*sizeAutomat + j), i, j);
    }
	
	// create states matrix
	GrB_Matrix States;
	info = GrB_Matrix_new(&States, GrB_INT64, sizeAutomat, sizeAutomat);
	
	if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
	
	for (int i = 0; i < sizeAutomat; i++)
	{
		for (int j = 0; j < sizeAutomat; j++)
			GrB_Matrix_setElement_INT64(States, vector_int_get_element_by_index(&grammar->states, i*sizeAutomat + j), i, j);
	}
	
    // create graph matrix
    GrB_Matrix Graph;
    uint64_t sizeGraph = Graph_RequiredMatrixDim(gc->g);
    info = GrB_Matrix_new(&Graph, GrB_INT64, sizeGraph, sizeGraph);
	
	if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");

    for (int i = 0; i < GraphContext_SchemaCount(gc, SCHEMA_EDGE); i++)
    {
        char *label = gc->relation_schemas[i]->name;
        int label_id = map_get_second_by_first(grammar->indices, label);

        for (int j = 0; j < sizeGraph; j++)
        {
            for (int k = 0; k < sizeGraph; k++)
            {
                bool has_element = false;
                GrB_Matrix_extractElement_BOOL(&has_element, gc->relation_schemas[i], j, k);
                if (has_element)
                {
                    int64_t oldelement = 0;
                    GrB_Matrix_extractElement_INT64(&oldelement, Graph, j, k);
                    GrB_Matrix_setElement_INT64(Graph, oldelement | label_id, j, k);
                }
            }
        }
    }

    // create binary operation
    GrB_BinaryOp binary_for_kron;
    GrB_BinaryOp_new(&binary_for_kron, F_BINARY(binary_for_kron), GrB_INT64, GrB_INT64, GrB_INT64);

    // create bool monoid for bool semiring
    GrB_Monoid bool_monoid;
    GrB_Monoid_new_BOOL(&bool_monoid, GrB_LOR, 0);

    // create bool semiring
    GrB_Semiring bool_semiring;
    GrB_Semiring_new(&bool_semiring, bool_monoid, GrB_LAND);

    // create matrix for kronecker product and transitive clouser
    GrB_Matrix Kproduct;
    uint32_t sizeKproduct = sizeGraph * sizeAutomat;
    info = GrB_Matrix_new(&Kproduct, GrB_INT64, sizeKproduct, sizeKproduct);
	
	if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
	
    GrB_Matrix Tclouser; // чтобы быстрее было добавлять новые дуги в граф
    info = GrB_Matrix_new(&Tclouser, GrB_BOOL, sizeKproduct, sizeKproduct);
	
	if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
	
    GrB_Matrix degreeKproduct;
    info = GrB_Matrix_new(&degreeKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
	
	if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");

    // algorithm
    bool matrices_is_changed = true;

    while(matrices_is_changed)
    {
        matrices_is_changed = false;
		response->iteration_count++;

        GxB_kron(Kproduct, NULL, NULL, binary_for_kron, Automat, Graph, NULL);

        // transitive clouser
        bool transitive_matrix_is_changed = true;

        GrB_Matrix_dup(&degreeKproduct, Kproduct);

        int32_t nvalsPrev = 0;
        int32_t nvalsCur = 0;
        while (transitive_matrix_is_changed)
        {
            GrB_mxm(degreeKproduct, NULL, NULL, bool_semiring, degreeKproduct, Kproduct, NULL);
            GrB_eWiseAdd_Matrix_BinaryOp(Tclouser, NULL, NULL, GrB_LOR, Kproduct, degreeKproduct, NULL);

            GrB_Matrix_nvals(&nvalsCur, Tclouser);
            if (nvalsCur == nvalsPrev)
                transitive_matrix_is_changed = false;
            else
                nvalsPrev = nvalsCur;
        }
        GrB_Matrix_free(&degreeKproduct);

        // update graph
        for (int i = 0; i < sizeKproduct; i++)
        {
            for (int j = 0; j < sizeKproduct; j++)
            {
                int32_t s = 0;
                GrB_Matrix_extractElement_INT64(&s, Tclouser, i, j);

                if (s != 0)
                {
                    int i_1 = i / sizeAutomat;
                    int j_1 = j / sizeGraph;

                    int i_2 = i % sizeAutomat;
                    int j_2 = j % sizeGraph;

                    int64_t st = 0;
                    GrB_Matrix_extractElement_INT64(&st, States, i_1, i_2);
                    if (st != 0)
                    {
                        int64_t data = 0;
                        GrB_Matrix_extractElement_INT64(&data, Graph, i_2, j_2);
                        int64_t newdata = data | st;
                        if (data != newdata)
                            matrices_is_changed = true;
                        GrB_Matrix_setElement_INT64(Graph, newdata, i_2, j_2);
                    }
                }
            }
        }

        GrB_Matrix_clear(Tclouser);
        GrB_Matrix_clear(degreeKproduct);
        GrB_Matrix_clear(Kproduct);
    }

    GrB_Semiring_free(&bool_semiring);
    GrB_Monoid_free(&bool_monoid);
    GrB_BinaryOp_free(&binary_for_kron);

    GrB_Matrix_free(&Kproduct);
    GrB_Matrix_free(&degreeKproduct);
    GrB_Matrix_free(&Tclouser);
    GrB_Matrix_free(&Graph);
	GrB_Matrix_free(&States);
    GrB_Matrix_free(&Automat);

    return REDISMODULE_OK;
}
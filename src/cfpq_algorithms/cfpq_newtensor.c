#include "cfpq_algorithms.h"
#include "../redismodule.h"
#include "../automat/automat.h"
#include "../util/mem_prof/mem_prof.h"
#include "../util/simple_timer.h"

#define F_BINARY(f) ((void (*)(void *, const void *, const void *)) f)

int64_t label_for_kron = 0;
void bin_for_kron_new(bool *z, const int64_t *x, const bool *y)
{
    if ((*x & label_for_kron) != 0 && *y)
        *z = true;
    else
        *z = false;
}

int CFPQ_tensor_new(RedisModuleCtx *ctx, GraphContext *gc, automat *grammar, CfpqResponse *response)
{
    MemInfo mem_start;
    mem_usage_tick(&mem_start);
    // create automat matrix

    GrB_init(GrB_BLOCKING);
    GrB_Matrix Automat;
    uint64_t sizeAutomat = grammar->statesCount;
    GrB_Info info = GrB_Matrix_new(&Automat, GrB_INT64, sizeAutomat, sizeAutomat);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");

    for (int64_t i = 0; i < sizeAutomat; i++)
    {
        for (int64_t j = 0; j < sizeAutomat; j++)
        {
            int64_t element = vector_int_get_element_by_index(&grammar->matrix, i * sizeAutomat + j);

            if (element != (int64_t) 0)
               GrB_Matrix_setElement_INT64(Automat, element, i, j); 
        }
    }

    // create states matrix
    GrB_Matrix States;
    info = GrB_Matrix_new(&States, GrB_INT64, sizeAutomat, sizeAutomat);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");

    for (uint64_t i = 0; i < sizeAutomat; i++)
    {
        for (uint64_t j = 0; j < sizeAutomat; j++)
        {
            int64_t element = vector_int_get_element_by_index(&grammar->states, i * sizeAutomat + j);

            if (element != (int64_t) 0)
               GrB_Matrix_setElement_INT64(States, element, i, j); 
        }    
    }

    // create graph matrix
	//GrB_Matrix Graph;
    //uint64_t sizeGraph = Graph_RequiredMatrixDim(gc->g);
    //info = GrB_Matrix_new(&Graph, GrB_INT64, sizeGraph, sizeGraph);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
    

    uint64_t count_matrices = GraphContext_SchemaCount(gc, SCHEMA_EDGE) / 2;
    uint64_t sizeGraph = Graph_RequiredMatrixDim(gc->g);
    GrB_Matrix matrices[count_matrices + 1];
    GrB_Vector map_id;
    GrB_Vector_new(&map_id, GrB_INT64, count_matrices + 1);

    for (uint64_t i = 0; i < 2 * count_matrices; i++)
    {
        if (i % 2 != 0)
        {
            GrB_eWiseAdd_Matrix_BinaryOp(matrices[i / 2], NULL, NULL, GrB_LOR, matrices[i / 2], gc->g->relations[i], NULL);
        }
        else
        {
            GrB_Matrix_new(&matrices[i / 2], GrB_BOOL, sizeGraph, sizeGraph);
            GrB_Matrix_dup(&matrices[i / 2], gc->g->relations[i]);
            char *label = gc->relation_schemas[i]->name;
            int64_t label_id = map_get_second_by_first(&grammar->indices, label);
            GrB_Vector_setElement_INT64(map_id, label_id, i);
        }
    }
    int64_t label_id = map_get_second_by_first(&grammar->indices, "S");
    GrB_Vector_setElement_INT64(map_id, label_id, count_matrices);
	GrB_Matrix_new(&matrices[count_matrices], GrB_BOOL, sizeGraph, sizeGraph);
    GxB_print(matrices[0], 3);
    GxB_print(matrices[1], 3);
    GxB_print(matrices[2], 3);
    printf("%ld", count_matrices);
    //printf("IIIII");
    // create binary operation
    GrB_BinaryOp binary_for_kron;
    GrB_Info inf = GrB_BinaryOp_new(&binary_for_kron, F_BINARY(bin_for_kron_new), GrB_BOOL, GrB_INT64, GrB_BOOL);
    assert(inf == GrB_SUCCESS);

    // create bool monoid for bool semiring
    GrB_Monoid monoid;
    inf = GrB_Monoid_new_BOOL(&monoid, GrB_LOR, false);
    assert(inf == GrB_SUCCESS && "-2");

    // create bool semiring
    GrB_Semiring semiring;
    GrB_Semiring bool_lor;
    inf = GrB_Semiring_new(&semiring, monoid, GrB_LAND); // for mxm
    assert(inf == GrB_SUCCESS && "-1");

    // create matrix for kronecker product and transitive clouser
    GrB_Matrix Kproduct;
    uint64_t sizeKproduct = sizeGraph * sizeAutomat;
    GrB_Matrix_new(&Kproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    GrB_Matrix degreeKproduct;
    GrB_Matrix_new(&degreeKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    GrB_Matrix block;
    GrB_Matrix_new(&block, GrB_BOOL, sizeGraph, sizeGraph);
    GrB_Matrix partKproduct;
	GrB_Matrix_new(&partKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    GrB_Matrix Kproduct_s;
    GrB_Matrix_new(&Kproduct_s, GrB_BOOL, sizeKproduct, sizeKproduct); 

    for (uint64_t i = 0; i < count_matrices; i++)
    {
        GrB_Vector_extractElement_INT64(&label_for_kron, map_id, i);
        GxB_kron(partKproduct, NULL, NULL, binary_for_kron, Automat, matrices[i], NULL);
        GrB_eWiseAdd_Matrix_BinaryOp(Kproduct_s, NULL, NULL, GrB_LOR, Kproduct_s, partKproduct, NULL);
        GrB_Matrix_clear(partKproduct);
    }

    // algorithm
    bool matrices_is_changed = true;
    GrB_Index nvals = 0;
    while(matrices_is_changed)
    {
        matrices_is_changed = false;
        response->iteration_count++;

        //inf = GxB_kron(Kproduct, NULL, NULL, binary_for_kron, Automat, Graph, NULL);
        //assert(inf == GrB_SUCCESS && "1");

        GrB_Vector_extractElement_INT64(&label_for_kron, map_id, count_matrices);
        GxB_kron(partKproduct, NULL, NULL, binary_for_kron, Automat, matrices[count_matrices], NULL);
        GrB_eWiseAdd_Matrix_BinaryOp(Kproduct, NULL, NULL, GrB_LOR, Kproduct_s, partKproduct, NULL);
        GrB_Matrix_clear(partKproduct);

        GxB_select(Kproduct, NULL, NULL, GxB_NONZERO, Kproduct, NULL, NULL);
        GxB_print(Kproduct, 3);
        // transitive clouser
		bool transitive_matrix_is_changed = true;

        info = GrB_Matrix_dup(&degreeKproduct, Kproduct);
        assert(info == GrB_SUCCESS && "2");

        uint64_t val_transitive = 0;
        uint64_t prev = 0;

        while (transitive_matrix_is_changed)
        {
            transitive_matrix_is_changed = false;

            info = GrB_mxm(degreeKproduct, NULL, NULL, semiring, degreeKproduct, Kproduct, NULL);
            assert(info == GrB_SUCCESS && "3");
            GxB_select(degreeKproduct, NULL, NULL, GxB_NONZERO, degreeKproduct, NULL, NULL);

            info = GrB_eWiseAdd_Matrix_BinaryOp(Kproduct, NULL, NULL, GrB_LOR, Kproduct, degreeKproduct, NULL);
            assert(info == GrB_SUCCESS && "4");

            GrB_Matrix_nvals(&val_transitive, Kproduct);

            if (val_transitive != prev)
            {
               transitive_matrix_is_changed = true;
               prev = val_transitive;
            }
        }

        // update graph

        for (uint64_t i = 0; i < sizeAutomat; i++)
		{
            for (uint64_t j = 0; j < sizeAutomat; j++)
            {
                int64_t st = 0;
                GrB_Matrix_extractElement_INT64(&st, States, i, j);
                if (st != 0)
                {
                   GrB_Index I[sizeGraph];
                   GrB_Index J[sizeGraph];
                   for (uint64_t k = 0; k < sizeGraph; k++)
                   {
                       I[k] = i * sizeGraph + k;
                       J[k] = j * sizeGraph + k;
                   }
                   GrB_Matrix_extract(block, NULL, NULL, Kproduct, I, sizeGraph, J, sizeGraph, NULL);

                   for (uint64_t k = 0; k < sizeGraph; k++)
                   {
                       for (uint64_t l = 0; l < sizeGraph; l++)
                       {
                           bool s = false;
                           GrB_Matrix_extractElement_BOOL(&s, block, k, l);

                           if (s)
                           {
                              bool data = false;
                              GrB_Matrix_extractElement_BOOL(&data, matrices[count_matrices], k, l);

                              if (!data)
                              {
                                 nvals+=2;
								 matrices_is_changed = true;
                                 GrB_Matrix_setElement_BOOL(matrices[count_matrices], true, k, l);
                                 GrB_Matrix_setElement_BOOL(matrices[count_matrices], true, l, k);
                              }
                           }
                       }
                   }
                }
            }

        } 

        GrB_Matrix_clear(degreeKproduct);
        GrB_Matrix_clear(Kproduct);
    }
   
    CfpqResponse_Append(response, "S", nvals); 

    GrB_Semiring_free(&semiring);
    GrB_Monoid_free(&monoid);
    GrB_BinaryOp_free(&binary_for_kron);
    GrB_Matrix_free(&block);
    GrB_Matrix_free(&Kproduct);
    GrB_Matrix_free(&degreeKproduct);
    //GrB_Matrix_free(&Graph);
    GrB_Matrix_free(&Automat);
    GrB_Matrix_free(&partKproduct);
    GrB_Vector_free(&map_id);

    for (uint64_t i = 0; i < count_matrices + 1; i++)
         GrB_Matrix_free(&matrices[i]);
	MemInfo mem_delta;
    mem_usage_tok(&mem_delta, mem_start);
    response->vms_dif = mem_delta.vms;
    response->rss_dif = mem_delta.rss;
    response->shared_dif = mem_delta.share;

    return REDISMODULE_OK;
}




			
	
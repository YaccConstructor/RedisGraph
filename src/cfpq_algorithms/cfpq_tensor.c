#include "cfpq_algorithms.h"
#include "../redismodule.h"
#include "../automat/automat.h"
#include "../util/mem_prof/mem_prof.h"

#define F_BINARY(f) ((void (*)(void *, const void *, const void *)) f)

void bin_for_kron(bool *z, const int64_t *x, const int64_t *y)
{
    if ((*x & *y) != 0)
        *z = true;
    else
        *z = false;
}

int CFPQ_tensor(RedisModuleCtx *ctx, GraphContext *gc, automat *grammar, CfpqResponse *response)
{
    MemInfo mem_start;
    mem_usage_tick(&mem_start);
	
    // create automat matrix
    GrB_Matrix Automat;
    uint64_t sizeAutomat = grammar->statesCount;
    GrB_Info info = GrB_Matrix_new(&Automat, GrB_INT64, sizeAutomat, sizeAutomat);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix Automat\n");

    for (uint64_t i = 0; i < sizeAutomat; i++)
    {
        for (uint64_t j = 0; j < sizeAutomat; j++)
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
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix States\n");

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
    GrB_Matrix Graph;
    uint64_t sizeGraph = Graph_RequiredMatrixDim(gc->g);
    info = GrB_Matrix_new(&Graph, GrB_INT64, sizeGraph, sizeGraph);

    if (info != GrB_SUCCESS)
        RedisModule_ReplyWithError(ctx, "failed to construct the matrix Graph\n");
	
    // очень узкое место
    for (uint32_t i = 0; i < GraphContext_SchemaCount(gc, SCHEMA_EDGE) / 2; i++)
    {
        char *label = gc->relation_schemas[2 * i]->name;
        int64_t label_id = map_get_second_by_first(&grammar->indices, label);
        if (label_id == INT_MIN)
           label_id = 0;

        for (uint64_t j = 0; j < sizeGraph; j++)
        {
            for (uint64_t k = 0; k < sizeGraph; k++)
            {
                bool has_element = false;
                GrB_Matrix_extractElement_BOOL(&has_element, gc->g->relations[2 * i], j, k);

                if (has_element && (label_id != (int64_t) 0))
                    GrB_Matrix_setElement_INT64(Graph, label_id, j, k);
            }
	}
    }
	
    // create binary operation
    GrB_BinaryOp binary_for_kron;
    GrB_Info inf = GrB_BinaryOp_new(&binary_for_kron, F_BINARY(bin_for_kron), GrB_BOOL, GrB_INT64, GrB_INT64);
    assert(inf == GrB_SUCCESS);

    // create bool monoid for bool semiring
    GrB_Monoid monoid;
    inf = GrB_Monoid_new_BOOL(&monoid, GrB_LOR, false);
    assert(inf == GrB_SUCCESS && "failed monoid");

    // create bool semiring
    GrB_Semiring semiring;
    GrB_Semiring bool_lor;
    inf = GrB_Semiring_new(&semiring, monoid, GrB_LAND);
    assert(inf == GrB_SUCCESS && "failed semiring");

    // create matrix for kronecker product and transitive clouser
    GrB_Matrix Kproduct;
    uint64_t sizeKproduct = sizeGraph * sizeAutomat;
    inf = GrB_Matrix_new(&Kproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    assert(inf == GrB_SUCCESS && "failed Kron matrix");
	
    GrB_Matrix degreeKproduct;
    inf = GrB_Matrix_new(&degreeKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    assert(inf == GrB_SUCCESS && "failed degree Kron matrix");

    // algorithm
    bool matrices_is_changed = true;
    GrB_Index nvals = 0;
    while(matrices_is_changed)
    {
        matrices_is_changed = false;
        response->iteration_count++;
		
	// calc Kron product
        GxB_kron(Kproduct, NULL, NULL, binary_for_kron, Automat, Graph, NULL);
        
	// delete zero elements
        GxB_select(Kproduct, NULL, NULL, GxB_NONZERO, Kproduct, NULL, NULL);

        // transitive clouser
	// tcK = sum[i = 0..n] (K^i)
        bool transitive_matrix_is_changed = true;

        GrB_Matrix_dup(&degreeKproduct, Kproduct);

        GrB_Index cur_elements = 0;  // for control
        GrB_Index prev_elements = 0; // 	changes
        while (transitive_matrix_is_changed)
        {
            transitive_matrix_is_changed = false;
			
	    // calc K^i
            GrB_mxm(degreeKproduct, NULL, NULL, semiring, degreeKproduct, Kproduct, NULL);
            
            // delete zero elements
            GxB_select(degreeKproduct, NULL, NULL, GxB_NONZERO, degreeKproduct, NULL, NULL);
			
            // add to tcK
            GrB_eWiseAdd_Matrix_BinaryOp(Kproduct, NULL, NULL, GrB_LOR, Kproduct, degreeKproduct, NULL);
			
	    // get current changes
            GrB_Matrix_nvals(&val_transitive, Kproduct);
            if (cur_elements != prev_elements)
            {
               transitive_matrix_is_changed = true;
               prev = val_transitive;
            }
        }
        
        // update graph
        for (uint64_t i = 0; i < sizeKproduct; i++)
        {
            for (uint64_t j = 0; j < sizeKproduct; j++)
            {
                bool element_Kproduct = false;
                GrB_Matrix_extractElement_BOOL(&element_Kproduct, Kproduct, i, j);

                if (element_Kproduct)
                {
                    int64_t sf = 0;
                    GrB_Matrix_extractElement_INT64(&sf, States, i / sizeGraph, j / sizeGraph);

                    if (sf != 0)
                    {
                        int64_t data = 0;
                        GrB_Matrix_extractElement_INT64(&data, Graph, i % sizeGraph, j % sizeGraph);
                        int64_t newdata = data | sf;

                        if (data != newdata)
			{
                            nvals++;
                            matrices_is_changed = true;
                            GrB_Matrix_setElement_INT64(Graph, newdata, i % sizeGraph, j % sizeGraph);
                            //GrB_Matrix_setElement_INT64(Graph, newdata, j % sizeGraph, i % sizeGraph);  не надо для RDF
                        }
                    }
                }
            }
        }
		
        GrB_Matrix_clear(degreeKproduct);
        GrB_Matrix_clear(Kproduct);
    }
	
    CfpqResponse_Append(response, "S", nvals);
	
    MemInfo mem_delta;
    mem_usage_tok(&mem_delta, mem_start);
    response->vms_dif = mem_delta.vms;
    response->rss_dif = mem_delta.rss;
    response->shared_dif = mem_delta.share;

    GrB_Semiring_free(&semiring);
    GrB_Monoid_free(&monoid);
    GrB_BinaryOp_free(&binary_for_kron);

    GrB_Matrix_free(&Kproduct);
    GrB_Matrix_free(&degreeKproduct);
    GrB_Matrix_free(&Graph);
    GrB_Matrix_free(&Automat);
    

    return REDISMODULE_OK;
}

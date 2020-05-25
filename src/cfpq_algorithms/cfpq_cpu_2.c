#include "cfpq_algorithms.h"
#include "../redismodule.h"
#include "../bool_automat/bool_automat.h"
#include "../util/mem_prof/mem_prof.h"
#include "../util/simple_timer.h"

GrB_Index I[30000000];
GrB_Index J[30000000];

int CFPQ_cpu2(RedisModuleCtx *ctx, GraphContext *gc, bool_automat *grammar, CfpqResponse *response)
{

    MemInfo mem_start;
    mem_usage_tick(&mem_start);

    // create automat matrix
    GrB_init(GrB_BLOCKING);

    int count_automat = grammar->count_matrices;
    uint64_t sizeAutomat = grammar->size_matrices;
    GrB_Matrix Automat[10];
    
    for (int i = 0; i < count_automat; i++)
    {
	GrB_Info info = GrB_Matrix_new(&Automat[i], GrB_BOOL, sizeAutomat, sizeAutomat);
	GxB_set(Automat[i], GxB_HYPER, GxB_HYPER_DEFAULT);
	if (info != GrB_SUCCESS)
            RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
    }

    for (int i = 0; i < count_automat; i++)
	GrB_Matrix_dup(&Automat[i], grammar->matrices[i]);


    // create states matrix
    int count_S_term = grammar->count_S_term;
    GrB_Matrix States[10];
    for (int i = 0; i < count_S_term; i++)
    {
	GrB_Info info = GrB_Matrix_new(&States[i], GrB_BOOL, sizeAutomat, sizeAutomat);
	GxB_set(States[i], GxB_HYPER, GxB_HYPER_DEFAULT);
	if (info != GrB_SUCCESS)
            RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
    }

    for (int i = 0; i < count_S_term; i++)
	GrB_Matrix_dup(&States[i], grammar->matrices[count_automat + i]);

    // create graph matrix
    uint64_t count_matrices = GraphContext_SchemaCount(gc, SCHEMA_EDGE)/2;
    uint64_t sizeGraph = Graph_RequiredMatrixDim(gc->g);
    GrB_Matrix matrices[10];

	for (uint64_t i = 0; i < count_matrices + count_S_term; i++)
	{
		GrB_Info info = GrB_Matrix_new(&matrices[i], GrB_BOOL, sizeGraph, sizeGraph);
		GxB_set(matrices[i], GxB_HYPER, GxB_HYPER_DEFAULT);
		if (info != GrB_SUCCESS)
        	    RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
	}

	for (uint64_t i = 0; i < 2*count_matrices; i+=2)
	{
		char *label = gc->relation_schemas[i]->name;
		int64_t label_id = map_get_second_by_first(&grammar->indices, label);
		uint64_t index = 0;

		   //assert(label_id!=INT_MIN);
		//printf("%s \n", label);
		if (label_id == INT_MIN)
		    continue;
		 printf("%s \n", label);
		while (grammar->id[index] != label_id && index < count_matrices)
		{
			index++;
		}

		//GrB_Matrix_dup(&matrices[index], gc->g->relations[i]);
		GrB_eWiseAdd_Matrix_BinaryOp(matrices[index], NULL, NULL, GrB_LOR, matrices[index], gc->g->relations[i], NULL);
		GrB_eWiseAdd_Matrix_BinaryOp(matrices[index], NULL, NULL, GrB_LOR, matrices[index], gc->g->relations[i + 1], NULL);
	}

//count_matrices = 2;


    // create bool monoid for bool semiring
    GrB_Monoid monoid;
    GrB_Info inf = GrB_Monoid_new_BOOL(&monoid, GrB_LOR, false);
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
    GxB_set(Kproduct, GxB_HYPER, GxB_HYPER_DEFAULT);
    GrB_Matrix degreeKproduct;
    GrB_Matrix_new(&degreeKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    GxB_set(degreeKproduct, GxB_HYPER, GxB_HYPER_DEFAULT);
    GrB_Matrix block;
    GrB_Matrix_new(&block, GrB_BOOL, sizeGraph, sizeGraph);
    GxB_set(block, GxB_HYPER, GxB_HYPER_DEFAULT);

    GrB_Info info;
	
    GrB_Matrix partKproduct;
    GrB_Matrix_new(&partKproduct, GrB_BOOL, sizeKproduct, sizeKproduct);
    GxB_set(partKproduct, GxB_HYPER, GxB_HYPER_DEFAULT);

    GrB_Matrix prevG;
    GrB_Matrix_new(&prevG, GrB_BOOL, sizeGraph, sizeGraph);
    GxB_set(prevG, GxB_HYPER, GxB_HYPER_DEFAULT);

    // algorithm
    bool matrices_is_changed = true;
    GrB_Index nvals = 0;
    while(matrices_is_changed)
    {
        matrices_is_changed = false;
        response->iteration_count++;
		
	for (uint64_t i = 0; i < count_matrices + count_S_term; i++)
	{
		GxB_kron(partKproduct, NULL, NULL, GrB_LAND, Automat[i], matrices[i], NULL);
		GrB_eWiseAdd_Matrix_BinaryOp(Kproduct, NULL, NULL, GrB_LOR, Kproduct, partKproduct, NULL);
		GrB_Matrix_clear(partKproduct);
	}

        GxB_select(Kproduct, NULL, NULL, GxB_NONZERO, Kproduct, NULL, NULL);

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
		
	for (int n = 0; n < count_S_term; n++)
	{
  	    for (uint64_t i = 0; i < sizeAutomat; i++)
	    {
            	for (uint64_t j = 0; j < sizeAutomat; j++)
            	{
            		bool st = false;
            		GrB_Matrix_extractElement_BOOL(&st, States[n], i, j);

            	if (st)
            	{
            	    for (uint64_t k = 0; k < sizeGraph; k++)
            	    {
            	        I[k] = i * sizeGraph + k;
            	        J[k] = j * sizeGraph + k;
					}

            	    GrB_Matrix_extract(block, NULL, NULL, Kproduct, I, sizeGraph, J, sizeGraph, NULL);
			
					int64_t id = grammar->id[count_matrices + n];
					int index = 0;
					while (grammar->id[index] != id)
					{
						index++;
					}
					GxB_select(block, NULL, NULL, GxB_NONZERO, block, NULL, NULL);
					GrB_Index val_check = 0;
					GrB_Matrix_nvals(&val_check, matrices[index]);
					GrB_Matrix_dup(&prevG, matrices[index]); 
					GrB_Matrix_apply(matrices[index], NULL, NULL, GxB_ONE_BOOL, block, NULL);
					GrB_eWiseAdd_Matrix_BinaryOp(matrices[index], NULL, NULL, GrB_LOR, prevG, matrices[index], NULL);
					GrB_Index nval_check = 0;
					GxB_select(matrices[index], NULL, NULL, GxB_NONZERO, matrices[index], NULL, NULL);

					GrB_Matrix_nvals(&nval_check, matrices[index]);

					if (val_check != nval_check)
					    matrices_is_changed = true;
					GrB_Matrix_nvals(&nvals, matrices[index]);
			
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
    GrB_Matrix_free(&block);
    GrB_Matrix_free(&Kproduct);
    GrB_Matrix_free(&degreeKproduct);
    GrB_Matrix_free(&partKproduct);
    GrB_Matrix_free(&prevG);

	for (uint64_t i = 0; i < count_automat; i++)
		GrB_Matrix_free(&Automat[i]);

    for (uint64_t i = 0; i < count_matrices + count_S_term; i++)
         GrB_Matrix_free(&matrices[i]);

    MemInfo mem_delta;
    mem_usage_tok(&mem_delta, mem_start);
    response->vms_dif = mem_delta.vms;
    response->rss_dif = mem_delta.rss;
    response->shared_dif = mem_delta.share;

    return REDISMODULE_OK;
}

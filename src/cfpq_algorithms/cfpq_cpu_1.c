#include "cfpq_algorithms.h"
#include "../grammar/item_mapper.h"

int CFPQ_cpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response) {

    // Create matrices
    uint64_t nonterm_count = grammar->nontermMapper.count;
    uint64_t graph_size = Graph_RequiredMatrixDim(gc->g);
    GrB_Matrix matrices[nonterm_count];

    for (uint64_t i = 0; i < nonterm_count; ++i) {
        GrB_Info info = GrB_Matrix_new(&matrices[i], GrB_BOOL, graph_size, graph_size);
        if (info != GrB_SUCCESS) {
            RedisModule_ReplyWithError(ctx, "failed to construct the matrix\n");
        }
    }

    // Initialize matrices
    for (int i = 0; i < GraphContext_SchemaCount(gc, SCHEMA_EDGE); i++) {
        char *terminal = gc->relation_schemas[i]->name;

        MapperIndex terminal_id = ItemMapper_GetPlaceIndex((ItemMapper *) &grammar->tokenMapper, terminal);
        if (terminal_id != grammar->tokenMapper.count) {
            for (int j = 0; j < grammar->simple_rules_count; j++) {
                SimpleRule *simpleRule = &grammar->simple_rules[j];
                if (simpleRule->r == terminal_id) {
                    GrB_Matrix_dup(&matrices[simpleRule->l], gc->g->relations[i]);
                }
            }
        }
    }

    // Create monoid and semiring
    GrB_Monoid monoid;
    GrB_Semiring semiring;

    GrB_Info info = GrB_Monoid_new_BOOL(&monoid, GrB_LOR, false);
    assert(info == GrB_SUCCESS && "GraphBlas: failed to construct the monoid\n");

    info = GrB_Semiring_new(&semiring, monoid, GrB_LAND);
    assert(info == GrB_SUCCESS && "GraphBlas: failed to construct the semiring\n");

    // Super-puper algorithm
    bool matrices_is_changed = true;
    while(matrices_is_changed) {
        matrices_is_changed = false;

        response->iteration_count++;

        for (int i = 0; i < grammar->complex_rules_count; ++i) {
            MapperIndex nonterm1 = grammar->complex_rules[i].l;
            MapperIndex nonterm2 = grammar->complex_rules[i].r1;
            MapperIndex nonterm3 = grammar->complex_rules[i].r2;

            GrB_Index nvals_new, nvals_old;
            GrB_Matrix_nvals(&nvals_old, matrices[nonterm1]);

            GrB_mxm(matrices[nonterm1], GrB_NULL, GrB_LOR, semiring,
                    matrices[nonterm2], matrices[nonterm3], GrB_NULL);

            GrB_Matrix_nvals(&nvals_new, matrices[nonterm1]);
            if (nvals_new != nvals_old) {
                matrices_is_changed = true;
            }
        }
    }

#ifdef DEBUG
    // Write to redis output full result
    {
        GrB_Index nvals = graph_size * graph_size;
        GrB_Index I[nvals];
        GrB_Index J[nvals];
        bool values[nvals];

        printf("graph size: %lu\n", graph_size);
        for (int i = 0; i < grammar->nontermMapper.count; i++) {
            printf("%s: ", ItemMapper_Map((ItemMapper *) &grammar->nontermMapper, i));
            GrB_Matrix_extractTuples(I, J, values, &nvals, matrices[i]);
            for (int j = 0; j < nvals; j++) {
                printf("(%lu, %lu) ", I[j], J[j]);
            }
            printf("\n");
        }
    }
#endif

    // clean and write response
    for (int i = 0; i < grammar->nontermMapper.count; i++) {
        GrB_Index nvals;
        char* nonterm;

        GrB_Matrix_nvals(&nvals, matrices[i]) ;
        nonterm = ItemMapper_Map((ItemMapper *) &grammar->nontermMapper, i);
        CfpqResponse_Append(response, nonterm, nvals);

        GrB_Matrix_free(&matrices[i]) ;
    }
    GrB_Semiring_free(&semiring);
    GrB_Monoid_free(&monoid);

    return REDISMODULE_OK;
}

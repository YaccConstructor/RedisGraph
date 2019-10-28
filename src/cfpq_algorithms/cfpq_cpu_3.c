#include <assert.h>

#include "cfpq_algorithms.h"
#include "../grammar/item_mapper.h"


int CFPQ_cpu3(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response) {
    // Create matrices
    uint64_t nonterm_count = grammar->nontermMapper.count;
    uint64_t graph_size = Graph_RequiredMatrixDim(gc->g);

    GrB_Matrix A_top[nonterm_count];
    GrB_Matrix B[nonterm_count];

    for (uint64_t i = 0; i < nonterm_count; ++i) {
        GrB_Info A_top_info = GrB_Matrix_new(&A_top[i], GrB_BOOL, graph_size, graph_size);
        GrB_Info B_info = GrB_Matrix_new(&B[i], GrB_BOOL, graph_size, graph_size);

        if (A_top_info != GrB_SUCCESS || B_info !=GrB_SUCCESS)
            return 1;
    }

    // Initialize A_top with simple rules, B with empty.
    for (int i = 0; i < GraphContext_SchemaCount(gc, SCHEMA_EDGE); i++) {
        char *terminal = gc->relation_schemas[i]->name;
        MapperIndex terminal_id = ItemMapper_GetPlaceIndex((ItemMapper *) &grammar->tokenMapper, terminal);

        if (terminal_id != grammar->tokenMapper.count) {
            for (int j = 0; j < grammar->simple_rules_count; j++) {
                SimpleRule *simpleRule = &grammar->simple_rules[j];

                if (simpleRule->r == terminal_id) {
                    GrB_eWiseAdd_Matrix_BinaryOp(A_top[simpleRule->l], NULL, NULL,
                                                 GrB_LOR, A_top[simpleRule->l], gc->g->relations[i], NULL);
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

    // Create descriptor
    GrB_Descriptor reversed_mask;
    GrB_Descriptor_new(&reversed_mask);
    GrB_Descriptor_set(reversed_mask, GrB_MASK, GrB_SCMP);

    // Super-puper algorithm
    bool matrices_is_changed = true;
    while(matrices_is_changed) {
        response->iteration_count++;

        // Initialize new A_top and B matrices
        GrB_Matrix A_new[nonterm_count];
        for (uint64_t i = 0; i < nonterm_count; ++i) {
            GrB_Matrix_new(&A_new[i], GrB_BOOL, graph_size, graph_size);
            GrB_Matrix_dup(&A_new[i], A_top[i]);
        }

        for (int i = 0; i < grammar->complex_rules_count; ++i) {
            MapperIndex nonterm_l = grammar->complex_rules[i].l;
            MapperIndex nonterm_r1 = grammar->complex_rules[i].r1;
            MapperIndex nonterm_r2 = grammar->complex_rules[i].r2;

            // create product matrices
            GrB_Matrix AB, BA, AA;
            GrB_Matrix_new(&AB, GrB_BOOL, graph_size, graph_size);
            GrB_Matrix_new(&BA, GrB_BOOL, graph_size, graph_size);
            GrB_Matrix_new(&AA, GrB_BOOL, graph_size, graph_size);

            // Compute product matrices
            GrB_mxm(AB, NULL, NULL, semiring,
                    A_top[nonterm_r1], B[nonterm_r2], NULL);
            GrB_mxm(BA, NULL, NULL, semiring,
                    B[nonterm_r1], A_top[nonterm_r2], NULL);
            GrB_mxm(AA, NULL, NULL, semiring,
                    A_top[nonterm_r1], A_top[nonterm_r2], NULL);

            // Compute total A_new
            GrB_eWiseAdd_Matrix_BinaryOp(A_top[nonterm_l], NULL, NULL, GrB_LOR, A_top[nonterm_l], AB, NULL);
            GrB_eWiseAdd_Matrix_BinaryOp(A_top[nonterm_l], NULL, NULL, GrB_LOR, A_top[nonterm_l], BA, NULL);
            GrB_eWiseAdd_Matrix_BinaryOp(A_top[nonterm_l], NULL, NULL, GrB_LOR, A_top[nonterm_l], AA, NULL);

            GrB_Matrix_clear(AB);
            GrB_Matrix_free(&AB);
            GrB_free(&AB);

            GrB_Matrix_clear(BA);
            GrB_Matrix_free(&BA);
            GrB_free(&BA);

            GrB_Matrix_clear(AA);
            GrB_Matrix_free(&AA);
            GrB_free(&AA);
        }

        // Compute new B
        for (int i = 0; i < nonterm_count; ++i) {
            GrB_eWiseAdd_Matrix_BinaryOp(B[i], NULL, NULL, GrB_LOR, B[i], A_new[i], NULL);
            GrB_Matrix_clear(A_new[i]);
        }


        // Check existing new elements and write result to next step
        matrices_is_changed = false;
        for (uint64_t i = 0; i < nonterm_count; ++i) {
            GrB_Matrix_dup(&A_new[i], A_top[i]);
            GrB_Matrix_clear(A_top[i]);

            GxB_select(A_top[i], B[i], NULL, GxB_NONZERO, A_new[i], NULL, reversed_mask);

            GrB_Matrix_free(&A_new[i]);
            GrB_free(&A_new[i]);

            GrB_Index nvals_new;
            GrB_Matrix_nvals(&nvals_new, A_top[i]);

            if (nvals_new != 0)
                matrices_is_changed = true;
        }
    }

    // clean and write response
    for (int i = 0; i < nonterm_count; i++) {

        char* nonterm;

        GrB_Index nvals_B, nvals_A_top;
        GrB_Matrix_nvals(&nvals_B, B[i]);
        GrB_Matrix_nvals(&nvals_A_top, A_top[i]);

        nonterm = ItemMapper_Map((ItemMapper *) &grammar->nontermMapper, i);
        CfpqResponse_Append(response, nonterm, nvals_B + nvals_A_top);

        GrB_Matrix_clear(B[i]);
        GrB_Matrix_free(&B[i]);
        GrB_free(&B[i]);

        GrB_Matrix_clear(A_top[i]);
        GrB_Matrix_free(&A_top[i]);
        GrB_free(&A_top[i]);
    }

    GrB_Semiring_free(&semiring);
    GrB_Monoid_free(&monoid);
    return 0;
}
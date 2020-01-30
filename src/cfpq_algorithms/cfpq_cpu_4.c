#include "cfpq_algorithms.h"
#include "../grammar/item_mapper.h"
#include "../grammar/helpers.h"
#include "index.h"


int CFPQ_cpu4(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response) {
    // create index type and operations
    GrB_Type IndexType;
    check_info(GrB_Type_new(&IndexType, sizeof(Index)));

    GrB_BinaryOp IndexType_Add, IndexType_Mul;
    check_info(GrB_BinaryOp_new(&IndexType_Add, PathIndex_Add, IndexType, IndexType, IndexType));
    check_info(GrB_BinaryOp_new(&IndexType_Mul, PathIndex_Mul, IndexType, IndexType, IndexType));

    GrB_Monoid IndexType_Monoid;
    GrB_Info info = GrB_Monoid_new(&IndexType_Monoid, IndexType_Add, (void *) &PathIndex_Identity);
    check_info(info);

    GrB_Semiring IndexType_Semiring;
    check_info(GrB_Semiring_new(&IndexType_Semiring, IndexType_Monoid, IndexType_Mul));


    // Create matrices
    uint64_t nonterm_count = grammar->nontermMapper.count;
    uint64_t graph_size = Graph_RequiredMatrixDim(gc->g);

    GrB_Matrix matrices[nonterm_count];

    for (uint64_t i = 0; i < nonterm_count; ++i) {
        check_info(GrB_Matrix_new(&matrices[i], IndexType, graph_size, graph_size));
    }

    for (int i = 0; i < GraphContext_SchemaCount(gc, SCHEMA_EDGE); i++) {
        char *terminal = gc->relation_schemas[i]->name;

        MapperIndex terminal_id = ItemMapper_GetPlaceIndex((ItemMapper *) &grammar->tokenMapper, terminal);
        if (terminal_id != grammar->tokenMapper.count) {
            for (int j = 0; j < grammar->simple_rules_count; j++) {
                SimpleRule *simpleRule = &grammar->simple_rules[j];
                if (simpleRule->r == terminal_id) {
                    PathIndex_MatrixInit(&matrices[simpleRule->l], &gc->g->relations[i]);
                }
            }
        }
    }

    bool matrices_is_changed = true;
    while(matrices_is_changed) {
        response->iteration_count++;
        matrices_is_changed = false;

        for (int i = 0; i < grammar->complex_rules_count; ++i) {
            MapperIndex nonterm1 = grammar->complex_rules[i].l;
            MapperIndex nonterm2 = grammar->complex_rules[i].r1;
            MapperIndex nonterm3 = grammar->complex_rules[i].r2;

            GrB_Matrix m_old;
            GrB_Matrix_dup(&m_old, matrices[nonterm1]);

            GrB_mxm(matrices[nonterm1], GrB_NULL, IndexType_Add, IndexType_Semiring,
                    matrices[nonterm2], matrices[nonterm3], GrB_NULL);

            GrB_Index nvals_new, nvals_old;
            GrB_Matrix_nvals(&nvals_new, matrices[nonterm1]);
            GrB_Matrix_nvals(&nvals_old, m_old);

            if (nvals_new != nvals_old) {
                matrices_is_changed = true;
            }

            GrB_Matrix_free(&m_old);
            GrB_free(&m_old);
        }
    }

    for (int i = 0; i < grammar->nontermMapper.count; i++) {
        GrB_Index nvals;
        char* nonterm;

        GrB_Matrix_nvals(&nvals, matrices[i]) ;
        nonterm = ItemMapper_Map((ItemMapper *) &grammar->nontermMapper, i);
        CfpqResponse_Append(response, nonterm, nvals);

        GrB_Matrix_free(&matrices[i]) ;
    }
    GrB_Semiring_free(&IndexType_Semiring);
    GrB_Monoid_free(&IndexType_Monoid);

    return REDISMODULE_OK;
}
#include "../cfpq_algorithms.h"
#include "../../grammar/item_mapper.h"
#include "../../grammar/helpers.h"
#include "../index.h"
#include "all_path_response.h"
#include "../../util/arr.h"
#include "../../util/simple_timer.h"


int CFPQ_cpu4(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response) {
    // Start index building timer
    double timer[2];
    simple_tic(timer);

    // create index type and operations
    GrB_Type IndexType;
    check_info(GrB_Type_new(&IndexType, sizeof(PathIndex)));

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

    // Algorithm
    bool matrices_is_changed = true;
    while(matrices_is_changed) {
        response->iteration_count++;
        matrices_is_changed = false;

        for (int i = 0; i < grammar->complex_rules_count; ++i) {
            MapperIndex nonterm1 = grammar->complex_rules[i].l;
            MapperIndex nonterm2 = grammar->complex_rules[i].r1;
            MapperIndex nonterm3 = grammar->complex_rules[i].r2;

            GrB_Index nvals_old;
            GrB_Matrix_nvals(&nvals_old, matrices[nonterm1]);

            GrB_mxm(matrices[nonterm1], GrB_NULL, IndexType_Add, IndexType_Semiring,
                    matrices[nonterm2], matrices[nonterm3], GrB_NULL);

            GrB_Index nvals_new;
            GrB_Matrix_nvals(&nvals_new, matrices[nonterm1]);

            if (nvals_new != nvals_old) {
                matrices_is_changed = true;
            }
        }
    }
    // Compute index building time
    double index_time = simple_toc(timer);

    // Iterate over nz pairs and find paths
    int start_nonterm = ItemMapper_GetPlaceIndex((ItemMapper*) &grammar->nontermMapper, "s");

    GxB_MatrixTupleIter *it;
    GxB_MatrixTupleIter_new(&it, matrices[start_nonterm]);

    GrB_Index left, right;
    bool depleted = false;

    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, matrices[start_nonterm]);

    // Create csv for path stat
    char s[200];
    sprintf(s, "%s.csv", gc->graph_name);

    FILE *f = fopen(s, "w");
    sprintf(s, "length,i,j,time");
    fputs(s, f);

    int i = 0;
    while (true) {
        i++;
        if (i % 100000 == 0) {
            printf("%d\n", i);
            fflush(stdout);
        }

        GxB_MatrixTupleIter_next(it, &left, &right, &depleted);
        if (depleted) {
            break;
        }

        simple_tic(timer);
        GrB_Index *path = PathIndex_MatrixGetPath(matrices, grammar, left, right, start_nonterm);
        if (path == NULL)
            continue;

        PathResponse item;
        item.length = array_len(path);
        item.left = left;
        item.right = right;
        item.time = simple_toc(timer);

        sprintf(s, "\n%d,%lu,%lu,%f", item.length, item.left, item.right, item.time);
        fputs(s, f);

        array_free(path);
    }
    fclose(f);
    AllPathResponse *allPathResponse = AllPathResponse_New(1);
    allPathResponse->index_time = index_time;
    response->customResp = (CustomResponseBase *) allPathResponse;

    // Free matrices
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
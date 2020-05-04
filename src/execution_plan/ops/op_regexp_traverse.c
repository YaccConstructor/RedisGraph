#include "op_regexp_traverse.h"
#include "../../query_ctx.h"
#include "../../arithmetic/algebraic_expression/utils.h"
#include "../../util/simple_timer.h"

static Record RegexpTraverseConsume(OpBase *opBase);
static OpResult RegexpTraverseReset(OpBase *ctx);
static void RegexpTraverseFree(OpBase *ctx);
static inline int RegexpTraverseToString(const OpBase *op_base, char *buf, uint buf_len);
void _regexp_traverse(RegexpTraverse *op);

OpBase *NewRegexpTraverseOp(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae) {
    RegexpTraverse *op = rm_calloc(1, sizeof(RegexpTraverse));
    op->plan = plan;
    op->graph = g;

    op->ae = ae;
    op->ae_m = NULL;

    op->F = GrB_NULL;
    op->M = GrB_NULL;
    op->iter = NULL;

    op->recordsCap = 32;
    op->recordsLen = 0;
    op->records = rm_calloc(op->recordsCap, sizeof(Record));
    op->r = NULL;

    op->deps = PathPatternCtx_GetDependencies(plan->path_pattern_ctx, ae);

#ifdef DEBUG_PATH_PATTERNS
    printf("Dependencies: ");
    for (int i = 0; i < array_len(op->deps); ++i) {
        printf("%s ", op->deps[i]->name);
        fflush(stdout);
    }
    printf("\n");
#endif

    OpBase_Init((OpBase *)op, OPType_REGEXP_TRAVERSE, "Regexp Traverse", NULL,
                RegexpTraverseConsume, RegexpTraverseReset, RegexpTraverseToString,
                NULL,RegexpTraverseFree, false, plan);

    assert(OpBase_Aware((OpBase *)op, AlgebraicExpression_Source(ae), &op->srcNodeIdx));
    op->destNodeIdx = OpBase_Modifies((OpBase *)op, AlgebraicExpression_Destination(ae));

    op->cnt = 0;
    return (OpBase *)op;
}

static Record RegexpTraverseConsume(OpBase *opBase) {
    RegexpTraverse *op = (RegexpTraverse*) opBase;
    OpBase *child = op->op.children[0];

//    op->cnt += 1;
//    printf("%d\n", op->cnt);
//    fflush(stdout);

    bool depleted = true;
    NodeID src_id = INVALID_ENTITY_ID;
    NodeID dest_id = INVALID_ENTITY_ID;

    while(true) {
        if(op->iter) GxB_MatrixTupleIter_next(op->iter, &src_id, &dest_id, &depleted);

        // Managed to get a tuple, break.
        if(!depleted) break;

        /* Run out of tuples, try to get new data.
         * Free old records. */
        op->r = NULL;
        for(int i = 0; i < op->recordsLen; i++) {
            OpBase_DeleteRecord(op->records[i]);
        }

        // Ask child operations for data.
        for(op->recordsLen = 0; op->recordsLen < op->recordsCap; op->recordsLen++) {
            Record childRecord = OpBase_Consume(child);
            // If the Record is NULL, the child has been depleted.
            if(!childRecord) break;

            // Store received record.
            Record_PersistScalars(childRecord);
            op->records[op->recordsLen] = childRecord;
        }

        // No data.
        if(op->recordsLen == 0) return NULL;

        _regexp_traverse(op);
    }

    /* Get node from current column. */
    op->r = op->records[src_id];
    Node destNode = {0};
    Graph_GetNode(op->graph, dest_id, &destNode);
    Record_AddNode(op->r, op->destNodeIdx, destNode);

    return OpBase_CloneRecord(op->r);
}

void _AlgebraicExpression_FetchReferences(AlgebraicExpression *exp, PathPatternCtx *pathPatternCtx) {
    switch(exp->type) {
        case AL_OPERATION: {
            uint child_count = AlgebraicExpression_ChildCount(exp);
            for (uint i = 0; i < child_count; i++) {
                _AlgebraicExpression_FetchReferences(CHILD_AT(exp, i), pathPatternCtx);
            }
            break;
        }
        case AL_OPERAND: {
            const char *reference = exp->operand.reference;
            if (exp->operand.reference != NULL) {
                PathPattern *pathPattern = PathPatternCtx_GetPathPattern(pathPatternCtx,reference);
                exp->operand.matrix = pathPattern->m;
            }
            break;
        }
        default:
            assert("Unknow algebraic expression node type" && false);
            break;
    }
}


static void _populate_filter_matrix(RegexpTraverse *op) {
    for(uint i = 0; i < op->recordsLen; i++) {
        Record r = op->records[i];
        /* Update filter matrix F, set row i at position srcId
         * F[i, srcId] = true. */
        Node *n = Record_GetNode(r, op->srcNodeIdx);
        NodeID srcId = ENTITY_GET_ID(n);
        GrB_Matrix_setElement_BOOL(op->F, true, i, srcId);
    }
}

void _path_pattern_traverse(RegexpTraverse *op) {
    bool is_changed;
    while (true) {
//		printf("here3.5\n");
//		fflush(stdout);
        is_changed = false;
        for (int i = 0; i < array_len(op->deps); ++i) {
            PathPattern *pattern = op->deps[i];
            AlgebraicExpression *pattern_expr = pattern->ae;

            _AlgebraicExpression_FetchReferences(pattern_expr, op->plan->path_pattern_ctx);
            GrB_Matrix m = AlgebraicExpression_EvalArbitrary(pattern_expr);

            GrB_Index nvals_old;
            GrB_Matrix_nvals(&nvals_old, pattern->m);
            GrB_Matrix_free(&pattern->m);

            pattern->m = m;
            GrB_Index nvals_new;
            GrB_Matrix_nvals(&nvals_new, pattern->m);

            if (nvals_old != nvals_new)
                is_changed = true;
        }
        if (!is_changed)
            break;
    }
}


void _regexp_traverse(RegexpTraverse *op) {
    // Precompute all Algebraic Expression result
    // and store it in ae_m.
//    printf("here1\n");
//    fflush(stdout);
    if (op->ae_m == NULL) {
        size_t required_dim = Graph_RequiredMatrixDim(op->graph);
        GrB_Matrix_new(&op->ae_m, GrB_BOOL, required_dim, required_dim);

        // Here would be execute cfpq algorithm...
//		printf("here3\n");
//		fflush(stdout);
        _path_pattern_traverse(op);
//		printf("here4\n");
//		fflush(stdout);
#ifdef DEBUG_PATH_PATTERNS
        for (int i = 0; i < array_len(op->deps); ++i) {
            PathPattern *pattern = op->deps[i];
            printf("Pattern: %s\n", PathPattern_ToString(pattern));
            GxB_print(pattern->m, GxB_COMPLETE);
        }
#endif

        _AlgebraicExpression_FetchReferences(op->ae, op->plan->path_pattern_ctx);
        op->ae_m = AlgebraicExpression_EvalArbitrary(op->ae);
//		printf("here5\n");
//		fflush(stdout);

#ifdef DEBUG_PATH_PATTERNS
        printf("_regexp_traverse ae_result = %s\n", AlgebraicExpression_ToStringDebug(op->ae));
        GxB_print(op->ae_m, GxB_COMPLETE);
#endif
    }
//	printf("here6\n");
//	fflush(stdout);

    // Create both filter and result matrices.
    if(op->F == GrB_NULL) {
        size_t required_dim = Graph_RequiredMatrixDim(op->graph);
        GrB_Matrix_new(&op->M, GrB_BOOL, op->recordsCap, required_dim);
        GrB_Matrix_new(&op->F, GrB_BOOL, op->recordsCap, required_dim);
    }
    _populate_filter_matrix(op);

    AlgebraicExpression *expr = AlgebraicExpression_NewOperand(op->ae_m, false, NULL, NULL, NULL, NULL, NULL);
    AlgebraicExpression_MultiplyToTheLeft(&expr, op->F);
    op->M = AlgebraicExpression_EvalArbitrary(expr);
    AlgebraicExpression_Free(expr);

    if(op->iter == NULL) GxB_MatrixTupleIter_new(&op->iter, op->M);
    else GxB_MatrixTupleIter_reuse(op->iter, op->M);

    GrB_Matrix_clear(op->F);
}

static OpResult RegexpTraverseReset(OpBase *ctx) {
    return OP_OK;
}

static void RegexpTraverseFree(OpBase *ctx) {
    RegexpTraverse *op = (RegexpTraverse *)ctx;
    if(op->iter) {
        GxB_MatrixTupleIter_free(op->iter);
        op->iter = NULL;
    }

    if(op->F != GrB_NULL) {
        GrB_Matrix_free(&op->F);
        op->F = GrB_NULL;
    }

    if(op->M != GrB_NULL) {
        GrB_Matrix_free(&op->M);
        op->M = GrB_NULL;
    }

    if(op->ae) {
        AlgebraicExpression_Free(op->ae);
        op->ae = NULL;
    }

    if(op->records) {
        for(int i = 0; i < op->recordsLen; i++) OpBase_DeleteRecord(op->records[i]);
        rm_free(op->records);
        op->records = NULL;
    }

    if (op->deps) {
        array_free(op->deps);
        op->deps = NULL;
    }

    if (op->ae_m != GrB_NULL) {
        GrB_Matrix_free(&op->ae_m);
        op->ae_m = NULL;
    }
}

static inline OpBase *CondTraverseClone(const ExecutionPlan *plan, const OpBase *opBase) {
    assert(opBase->type == OPType_CONDITIONAL_TRAVERSE);
    RegexpTraverse *op = (RegexpTraverse *)opBase;
    return NewRegexpTraverseOp(plan, QueryCtx_GetGraph(), AlgebraicExpression_Clone(op->ae));
}

static inline int RegexpTraverseToString(const OpBase *op_base, char *buf, uint buf_len) {
    RegexpTraverse *op = (RegexpTraverse *) op_base;

    int offset = 0;
    offset += snprintf(buf + strlen(buf), buf_len, "%s | %s -> %s",
            op_base->name, AlgebraicExpression_Source(op->ae), AlgebraicExpression_Destination(op->ae));

    return offset;
}

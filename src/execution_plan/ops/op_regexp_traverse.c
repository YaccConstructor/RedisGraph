#include "op_regexp_traverse.h"
#include "../../arithmetic/algebraic_expression.h"
#include "../../query_ctx.h"

//#define DEBUG_PATH_PATTERNS

static Record RegexpTraverseConsume(OpBase *opBase);
static OpResult RegexpTraverseReset(OpBase *ctx);
static void RegexpTraverseFree(OpBase *ctx);
static inline int RegexpTraverseToString(const OpBase *op_base, char *buf, uint buf_len);
void _regexp_traverse(RegexpTraverse *op);

OpBase *NewRegexpTraverseOp(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae) {
    RegexpTraverse *op = rm_calloc(1, sizeof(RegexpTraverse));
    op->graph = g;
//    op->deps = deps;

    op->ae = ae;
    op->ae_m = NULL;

    op->F = GrB_NULL;
    op->M = GrB_NULL;
    op->iter = NULL;

    op->recordsCap = 32;
    op->recordsLen = 0;
    op->records = rm_calloc(op->recordsCap, sizeof(Record));
    op->r = NULL;

    // Set our Op operations
    // TODO: OPType_CONDITIONAL_TRAVERSE
    OpBase_Init((OpBase *)op, OPType_REGEXP_TRAVERSE, "Regexp Traverse", NULL,
                RegexpTraverseConsume, RegexpTraverseReset, RegexpTraverseToString,
                NULL,RegexpTraverseFree, false, plan);

    assert(OpBase_Aware((OpBase *)op, AlgebraicExpression_Source(ae), &op->srcNodeIdx));
    op->destNodeIdx = OpBase_Modifies((OpBase *)op, AlgebraicExpression_Destination(ae));

    return (OpBase *)op;
}

static Record RegexpTraverseConsume(OpBase *opBase) {
    RegexpTraverse *op = (RegexpTraverse*) opBase;
    OpBase *child = op->op.children[0];

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
            if(!childRecord) break;

            // Store received record.
            op->records[op->recordsLen] = childRecord;
        }

        // No data.
        if(op->recordsLen == 0) return NULL;

        _regexp_traverse(op);
    }

    /* Get node from current column. */
    op->r = op->records[src_id];
    Node *destNode = Record_GetNode(op->r, op->destNodeIdx);
    Graph_GetNode(op->graph, dest_id, destNode);

    return OpBase_CloneRecord(op->r);
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

//void _path_pattern_traverse(RegexpTraverse *op) {
//    size_t required_dim = Graph_RequiredMatrixDim(op->graph);
//
//    bool is_changed;
//    while (true) {
//        is_changed = false;
//        for (int i = 0; i < array_len(op->deps); ++i) {
//            PathPattern *pattern = op->deps[i];
//            AlgebraicExpression *pattern_expr = pattern->ae_expr;
//            AlgebraicExpression_FetchOperands(pattern_expr);
//
//            GrB_Matrix m;
//            GrB_Matrix_new(&m, GrB_BOOL, required_dim, required_dim);
//            AlgebraicExpression_EvalArbitrary(pattern_expr, m);
//
//            GrB_Index nvals_old;
//            GrB_Matrix_nvals(&nvals_old, pattern->m);
//
//            GrB_Matrix_free(&pattern->m);
//            pattern->m = m;
//
//            GrB_Index nvals_new;
//            GrB_Matrix_nvals(&nvals_new, pattern->m);
//
//            if (nvals_old != nvals_new)
//                is_changed = true;
//        }
//        if (!is_changed)
//            break;
//    }
//}


void _regexp_traverse(RegexpTraverse *op) {
    // Precompute all Algebraic Expression result
    // and store it in ae_m.
    if (op->ae_m == NULL) {
        size_t required_dim = Graph_RequiredMatrixDim(op->graph);
        GrB_Matrix_new(&op->ae_m, GrB_BOOL, required_dim, required_dim);

        // Here would be execute cfpq algorithm...
//        _path_pattern_traverse(op);

//        for (int i = 0; i < array_len(op->deps); ++i) {
//            PathPattern *pattern = op->deps[i];
//            printf("Pattern %s = %s:\n", pattern->name, AlgebraicExpression_ToString(pattern->ae_expr));
//            GxB_print(pattern->m, GxB_COMPLETE);
//        }

        if (op->ae->type == AL_OPERAND) {
            GrB_Matrix_dup(&op->ae_m, op->ae->operand.matrix);
        } else {
            AlgebraicExpression_EvalArbitrary(op->ae, op->ae_m);
        }
#ifdef DEBUG_PATH_PATTERNS
        printf("_regexp_traverse ae_result = %s\n", AlgebraicExpression_ToString(op->ae));
        GxB_print(op->ae_m, GxB_COMPLETE);
#endif
    }

    // Create both filter and result matrices.
    if(op->F == GrB_NULL) {
        size_t required_dim = Graph_RequiredMatrixDim(op->graph);
        GrB_Matrix_new(&op->M, GrB_BOOL, op->recordsCap, required_dim);
        GrB_Matrix_new(&op->F, GrB_BOOL, op->recordsCap, required_dim);
    }
    _populate_filter_matrix(op);

    AlgebraicExpression *expr = AlgebraicExpression_NewOperand(op->ae_m, false, NULL, NULL, NULL, NULL);
    AlgebraicExpression_MultiplyToTheLeft(&expr, op->F);
    AlgebraicExpression_EvalArbitrary(expr, op->M);
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

//    offset += snprintf(buf + strlen(buf), buf_len, ": [");
//    for (int i = 0; i < array_len(op->deps); ++i) {
//        offset += snprintf(buf + strlen(buf), buf_len, "%s", op->deps[i]->name);
//    }
//    offset += snprintf(buf + strlen(buf), buf_len, "]");

    return offset;
}

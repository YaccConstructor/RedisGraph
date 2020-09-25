/*
* Copyright 2018-2020 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "op_conditional_traverse_dev.h"
#include "shared/print_functions.h"
#include "../../query_ctx.h"
#include "../../arithmetic/algebraic_expression/algebraic_expression_eval_dev.h"
#include "../../config.h"
//#define DPP

/* Forward declarations. */
static OpResult CondTraverseDevInit(OpBase *opBase);
static Record CondTraverseDevConsume(OpBase *opBase);
static OpResult CondTraverseDevReset(OpBase *opBase);
static OpBase *CondTraverseDevClone(const ExecutionPlan *plan, const OpBase *opBase);
static void CondTraverseDevFree(OpBase *opBase);

static int CondTraverseDevToString(const OpBase *ctx, char *buf, uint buf_len) {
	int offset = 0;
	offset += snprintf(buf, buf_len, "%s", ctx->name);
	return offset;
}

static void _populate_filter_matrix_dev(CondTraverseDev *op) {
	for(uint i = 0; i < op->recordCount; i++) {
		Record r = op->records[i];
		/* Update filter matrix F, set row i at position srcId
		 * F[i, srcId] = true. */
		Node *n = Record_GetNode(r, op->srcNodeIdx);
		NodeID srcId = ENTITY_GET_ID(n);
		GrB_Matrix_setElement_BOOL(op->F, true, i, srcId);
	}
}

static void _transitive_closure(PathPattern **deps, PathPatternCtx *pathPatternCtx, CondTraverseDev *op) {
#ifdef DPP
	printf("--------------before trans closure----------------\n");
	PathPatternCtx_Show(op->pathPatternCtx);
	for (int i = 0; i < array_len(deps); ++i) {
		PathPattern *p = deps[i];
		printf("Path Pattern %s:\n", p->reference.name);
		AlgebraicExpression_TotalShow(p->ae);
	}
#endif
	bool changed = true;

	size_t deps_size = array_len(deps);
	GrB_Index *nvals_ms = array_new(GrB_Index, deps_size);
	GrB_Index *nvals_srcs = array_new(GrB_Index, deps_size);

	GrB_Matrix *tmps = array_new(GrB_Matrix, deps_size);
	for (int i = 0; i < deps_size; ++i) {
		GrB_Index nrows, ncols;
		GrB_Matrix_nrows(&nrows, deps[i]->m);
		GrB_Matrix_ncols(&ncols, deps[i]->m);
		GrB_Matrix_new(&tmps[i], GrB_BOOL, nrows, ncols);
	}

#ifdef DPP
	int iter = 0;
#endif
	while (changed) {
		changed = false;
		for (int i = 0; i < deps_size; ++i) {
			GrB_Matrix_nvals(&nvals_ms[i], deps[i]->m);
			GrB_Matrix_nvals(&nvals_srcs[i], deps[i]->src);
		}

		for (int i = 0; i < array_len(deps); ++i) {
			PathPattern *pattern = deps[i];
			AlgebraicExpression_Eval_Dev(pattern->ae, tmps[i], pathPatternCtx);
			GrB_eWiseAdd_Matrix_BinaryOp(pattern->m, NULL, NULL, GrB_LOR, pattern->m, tmps[i], NULL);
		}

#ifdef DPP
		iter++;
		printf("------------------iter %d---------------------:\n", iter);
		for (int i = 0; i < array_len(deps); ++i) {
			AlgebraicExpression_TotalShow(deps[i]->ae);
		}
#endif

		for (int i = 0; i < deps_size; ++i) {
			GrB_Index nvals_m_new, nvals_src_new;
			GrB_Matrix_nvals(&nvals_m_new, deps[i]->m);
			GrB_Matrix_nvals(&nvals_src_new, deps[i]->src);

			if (nvals_ms[i] != nvals_m_new || nvals_srcs[i] != nvals_src_new) {
				changed = true;
			}
		}
	}

	for (int j = 0; j < deps_size; ++j) {
		GrB_Matrix_free(&tmps[j]);
	}
	array_free(tmps);
	array_free(nvals_srcs);
	array_free(nvals_ms);
#ifdef DPP
	printf("----after trans closure\n");
	for (int i = 0; i < array_len(deps); ++i) {
		PathPattern *p = deps[i];
		printf("PathPattern %s:\n", p->reference.name);
		printf("Src:\n");
		GxB_print(p->src, GxB_COMPLETE);
		printf("M:\n");
		GxB_print(p->m, GxB_COMPLETE);
	}
#endif
}

/* Evaluate algebraic expression:
 * prepends filter matrix as the left most operand
 * perform multiplications
 * set iterator over result matrix
 * removed filter matrix from original expression
 * clears filter matrix. */
void _traverse_dev(CondTraverseDev *op) {
	// If op->F is null, this is the first time we are traversing.
	if(op->F == GrB_NULL) {
#ifdef DPP
		printf("---------------\n");
		printf("First traverse:\n");
#endif
		// Create both filter and result matrices.
		size_t required_dim = Graph_RequiredMatrixDim(op->graph);
		GrB_Matrix_new(&op->M, GrB_BOOL, op->recordsCap, required_dim);
		GrB_Matrix_new(&op->F, GrB_BOOL, op->recordsCap, required_dim);

		// Prepend the filter matrix to algebraic expression as the leftmost operand.
		AlgebraicExpression_MultiplyToTheLeft(&op->ae, op->F);

		// Optimize the expression tree.
		AlgebraicExpression_Optimize(&op->ae);
		AlgebraicExpression_ReplaceTransposedReferences(op->ae);
		op->deps = PathPatternCtx_GetDependencies(op->pathPatternCtx, op->ae);

#ifdef DPP
		printf("Deps: ");
		for (int i = 0; i < array_len(op->deps); ++i) {
			printf("%s ", op->deps[i]->reference.name);
		}
		printf("\n");
#endif

		// Populate algebraic operand references with named path pattern matrices
		AlgebraicExpression_PopulateReferences(op->ae, op->pathPatternCtx);
		for (int i = 0; i < array_len(op->deps); ++i) {
			AlgebraicExpression_PopulateReferences(op->deps[i]->ae, op->pathPatternCtx);
		}

#ifdef DPP
		printf("AlgExp after optimize and populate: %s\n", AlgebraicExpression_ToStringDebug(op->ae));
		printf("---------------\n");
#endif
	}

	// Populate filter matrix.
	_populate_filter_matrix_dev(op);

#ifdef DPP
	printf("Filter matrix:\n");
	GxB_print(op->F, GxB_COMPLETE);
#endif

	// Clear named path patterns matrices
	PathPatternCtx_ClearMatrices(op->pathPatternCtx);

	// Evaluate expression for construct sources
	AlgebraicExpression_Eval_Dev(op->ae, op->M, op->pathPatternCtx);

#ifdef DPP
	printf("Result M before trans:\n");
	GxB_print(op->M, GxB_COMPLETE);
#endif

	// Perform transitive closure of named path patterns
	_transitive_closure(op->deps, op->pathPatternCtx, op);

	// Evaluate expression.
	AlgebraicExpression_Eval_Dev(op->ae, op->M, op->pathPatternCtx);

#ifdef DPP
	printf("Result M after trans:\n");
	GxB_print(op->M, GxB_COMPLETE);
#endif

	if(op->iter == NULL) GxB_MatrixTupleIter_new(&op->iter, op->M);
	else GxB_MatrixTupleIter_reuse(op->iter, op->M);

	// Clear filter matrix.
	GrB_Matrix_clear(op->F);
}

OpBase *NewCondTraverseDevOp(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae) {
#ifdef DPP
	printf("---------------------\n");
	printf("NewCondTraverseDevOp:\n");
	printf("AlgExp: %s\n", AlgebraicExpression_ToStringDebug(ae));
	PathPatternCtx_Show(plan->path_pattern_ctx);
	printf("---------------------\n");
#endif

	CondTraverseDev *op = rm_malloc(sizeof(CondTraverseDev));
	op->graph = g;
	op->ae = ae;

	op->pathPatternCtx = plan->path_pattern_ctx;

	op->r = NULL;
	op->iter = NULL;

	op->F = GrB_NULL;
	op->M = GrB_NULL;

	op->records = NULL;
	op->recordsCap = 0;
	op->recordCount = 0;

	op->dest_label = NULL;
	op->dest_label_id = GRAPH_NO_LABEL;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_CONDITIONAL_TRAVERSE, "Conditional Traverse", CondTraverseDevInit,
				CondTraverseDevConsume, CondTraverseDevReset, CondTraverseDevToString, CondTraverseDevClone, CondTraverseDevFree,
				false, plan);

	assert(OpBase_Aware((OpBase *)op, AlgebraicExpression_Source(ae), &op->srcNodeIdx));
	const char *dest = AlgebraicExpression_Destination(ae);
	op->destNodeIdx = OpBase_Modifies((OpBase *)op, dest);

	// Check the QueryGraph node and retrieve label data if possible.
	QGNode *dest_node = QueryGraph_GetNodeByAlias(plan->query_graph, dest);
	op->dest_label = dest_node->label;
	op->dest_label_id = dest_node->labelID;

	return (OpBase *)op;
}

static OpResult CondTraverseDevInit(OpBase *opBase) {
	CondTraverseDev *op = (CondTraverseDev *)opBase;
	op->recordsCap = Config_GetCfpqTraverseBufSize();
	op->records = rm_calloc(op->recordsCap, sizeof(Record));
	return OP_OK;
}

/* Each call to CondTraverseConsume emits a Record containing the
 * traversal's endpoints and, if required, an edge.
 * Returns NULL once all traversals have been performed. */
static Record CondTraverseDevConsume(OpBase *opBase) {
	CondTraverseDev *op = (CondTraverseDev *)opBase;
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
		for(uint i = 0; i < op->recordCount; i++) OpBase_DeleteRecord(op->records[i]);

		// Ask child operations for data.
		for(op->recordCount = 0; op->recordCount < op->recordsCap; op->recordCount++) {
			Record childRecord = OpBase_Consume(child);
			// If the Record is NULL, the child has been depleted.
			if(!childRecord) break;
			if(!Record_GetNode(childRecord, op->srcNodeIdx)) {
				/* The child Record may not contain the source node in scenarios like
				 * a failed OPTIONAL MATCH. In this case, delete the Record and try again. */
				OpBase_DeleteRecord(childRecord);
				op->recordCount--;
				continue;
			}

			// Store received record.
			Record_PersistScalars(childRecord);
			op->records[op->recordCount] = childRecord;
		}

		// No data.
		if(op->recordCount == 0) return NULL;

		_traverse_dev(op);
	}

	/* Get node from current column. */
	op->r = op->records[src_id];
	/* Populate the destination node and add it to the Record.
	 * Note that if the node's label is unknown, this will correctly
	 * create an unlabeled node. */
	Node destNode = GE_NEW_LABELED_NODE(op->dest_label, op->dest_label_id);
	Graph_GetNode(op->graph, dest_id, &destNode);
	Record_AddNode(op->r, op->destNodeIdx, destNode);

	return OpBase_CloneRecord(op->r);
}

static OpResult CondTraverseDevReset(OpBase *ctx) {
	CondTraverseDev *op = (CondTraverseDev *)ctx;

	// Do not explicitly free op->r, as the same pointer is also held
	// in the op->records array and as such will be freed there.
	op->r = NULL;
	for(uint i = 0; i < op->recordCount; i++) OpBase_DeleteRecord(op->records[i]);
	op->recordCount = 0;


	if(op->iter) {
		GxB_MatrixTupleIter_free(op->iter);
		op->iter = NULL;
	}
	if(op->F != GrB_NULL) GrB_Matrix_clear(op->F);
	return OP_OK;
}

static inline OpBase *CondTraverseDevClone(const ExecutionPlan *plan, const OpBase *opBase) {
	assert(opBase->type == OPType_CONDITIONAL_TRAVERSE);
	CondTraverseDev *op = (CondTraverseDev *)opBase;
	return NewCondTraverseDevOp(plan, QueryCtx_GetGraph(), AlgebraicExpression_Clone(op->ae));
}

/* Frees CondTraverse */
static void CondTraverseDevFree(OpBase *ctx) {
	CondTraverseDev *op = (CondTraverseDev *)ctx;
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
		for(uint i = 0; i < op->recordCount; i++) OpBase_DeleteRecord(op->records[i]);
		rm_free(op->records);
		op->records = NULL;
	}

	if (op->deps) {
		array_free(op->deps);
	}
}


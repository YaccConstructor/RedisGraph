#pragma once

#include "op.h"
#include "../../arithmetic/algebraic_expression.h"
//#include "../../path_patterns/path_pattern.h"
#include "../execution_plan.h"

/* OP Traverse */
typedef struct {
    OpBase op;
    Graph *graph;
//    PathPattern **deps;

    AlgebraicExpression *ae;    //
    GrB_Matrix ae_m;       // Matrix for precomputed alg exp.

    int srcNodeIdx;             // Index into record.
    int destNodeIdx;            // Index into record.

    GrB_Matrix F;               // Filter matrix.
    GrB_Matrix M;               // Algebraic expression result for records.
    GxB_MatrixTupleIter *iter;  // Iterator over M.

    int recordsCap;             // Max number of records to process.
    int recordsLen;             // Number of records to process.
    Record *records;            // Array of records.
    Record r;                   // Current selected record.

} RegexpTraverse;

/* Creates a new Traverse operation */
OpBase *NewRegexpTraverseOp(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae);

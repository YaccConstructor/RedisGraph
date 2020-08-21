#pragma once

#include "../algebraic_expression.h"
#include "../../path_patterns/path_pattern_ctx.h"

void AlgebraicExpression_Eval_Dev
(
	const AlgebraicExpression *exp, // Root node.
	GrB_Matrix res,                  // Result output.
	PathPatternCtx *pathCtx
);

void AlgebraicExpression_PopulateReferences(
	AlgebraicExpression *exp,
	PathPatternCtx *pathPatternCtx
);

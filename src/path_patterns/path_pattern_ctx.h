#pragma once

#include "path_pattern.h"

typedef struct {
    PathPattern **patterns;
} PathPatternCtx;

PathPatternCtx* PathPatternCtx_New();

void PathPatternCtx_AddPathPattern(PathPatternCtx *ctx, PathPattern *pattern);

PathPattern* PathPatternCtx_GetPathPattern(PathPatternCtx *ctx, const char* name);

/* Find all path pattern, that is reached from ebnf_expr (may be recursively)*/
PathPattern** PathPatternCtx_GetDependencies(PathPatternCtx *ctx, AlgebraicExpression *ebnf_expr);

PathPatternCtx *BuildPathPatternCtx(AST *ast, size_t required_dim);
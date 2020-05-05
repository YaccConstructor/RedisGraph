#pragma once

#include "path_pattern.h"

typedef struct {
    PathPattern **patterns;

	size_t required_matrix_dim;
    int anon_patterns_cnt;
} PathPatternCtx;

PathPatternCtx *PathPatternCtx_New(size_t required_matrix_dim);

void PathPatternCtx_Free(PathPatternCtx *pathPatternCtx);

void PathPatternCtx_AddPathPattern(PathPatternCtx *ctx, PathPattern *pattern);

const char *PathPatternCtx_GetNextAnonName(PathPatternCtx *ctx);

PathPattern* PathPatternCtx_GetPathPattern(PathPatternCtx *ctx, const char* name);

/* Find all path pattern, that is reached from expr (may be recursively)*/
PathPattern** PathPatternCtx_GetDependencies(PathPatternCtx *ctx, AlgebraicExpression *expr);

void PathPatternCtx_Show(PathPatternCtx *pathPatternCtx);
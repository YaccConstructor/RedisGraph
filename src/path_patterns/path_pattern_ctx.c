#include "path_pattern_ctx.h"
#include "../util/arr.h"
#include "../arithmetic/algebraic_expression/utils.h"

PathPattern* _PathPatternCtx_FindPathPattern(PathPattern **patterns, const char* name) {
    for (int i = 0; i < array_len(patterns); ++i) {
        PathPattern *pattern = patterns[i];
        if (strcmp(pattern->name, name) == 0)
            return pattern;
    }
    return NULL;
}

void _PathPatternCtx_DFS(PathPatternCtx *ctx, AlgebraicExpression *root, PathPattern ***visited) {
    if (root->type == AL_OPERATION) {
        for (int i = 0; i < AlgebraicExpression_ChildCount(root); ++i) {
            _PathPatternCtx_DFS(ctx, CHILD_AT(root, i), visited);
        }
    } else {
        const char *reference = root->operand.reference;
        if (reference != NULL && (_PathPatternCtx_FindPathPattern(*visited, reference) == NULL)) {
            PathPattern *next = PathPatternCtx_GetPathPattern(ctx, reference);
            assert(next != NULL && "Unresolved path reference");

            *visited = array_append(*visited, next);
            _PathPatternCtx_DFS(ctx, next->ae, visited);
        }
    }
}

PathPatternCtx *PathPatternCtx_New(size_t required_matrix_dim) {
    PathPatternCtx *ctx = rm_malloc(sizeof(PathPatternCtx));
    ctx->patterns = array_new(PathPattern *, 1);
	ctx->required_matrix_dim = required_matrix_dim;
    ctx->anon_patterns_cnt = 0;
    return ctx;
}

void PathPatternCtx_AddPathPattern(PathPatternCtx *ctx, PathPattern *pattern) {
    ctx->patterns = array_append(ctx->patterns, pattern);
}

const char *PathPatternCtx_GetNextAnonName(PathPatternCtx *ctx) {
	char *name = rm_malloc(10 * sizeof(char));
	sprintf(name, "anon_%d", ctx->anon_patterns_cnt++);
	return name;
}

PathPattern* PathPatternCtx_GetPathPattern(PathPatternCtx *ctx, const char* name) {
    return _PathPatternCtx_FindPathPattern(ctx->patterns, name);
}

PathPattern **PathPatternCtx_GetDependencies(PathPatternCtx *ctx, AlgebraicExpression *expr) {
    PathPattern **visited = array_new(PathPattern*,1);
    _PathPatternCtx_DFS(ctx, expr, &visited);
    return visited;
}

void PathPatternCtx_Free(PathPatternCtx *pathPatternCtx) {
    if (pathPatternCtx != NULL) {
        for (int i = 0; i < array_len(pathPatternCtx->patterns); ++i) {
            PathPattern_Free(pathPatternCtx->patterns[i]);
        }
        array_free(pathPatternCtx->patterns);
        rm_free(pathPatternCtx);
    }
}

void PathPatternCtx_Show(PathPatternCtx *pathPatternCtx) {
	printf("PathPatternCtx: [%d]\n", array_len(pathPatternCtx->patterns));
	for (int i = 0; i < array_len(pathPatternCtx->patterns); ++i) {
		printf("PATH PATTERN %s, %s, %s\n", pathPatternCtx->patterns[i]->name,
				EBNFBase_ToStr(pathPatternCtx->patterns[i]->ebnf_root),
				AlgebraicExpression_ToStringDebug(pathPatternCtx->patterns[i]->ae));
	}
	printf("----------------\n");
}

PathPatternCtx *PathPatternCtx_Clone(PathPatternCtx *pathCtx) {
    PathPatternCtx *clone = PathPatternCtx_New(pathCtx->required_matrix_dim);
    clone->anon_patterns_cnt = pathCtx->anon_patterns_cnt;
    for (int i = 0; i < array_len(pathCtx->patterns); ++i) {
        clone->patterns = array_append(clone->patterns,
									   PathPattern_Clone(pathCtx->patterns[i]));
    }
    return clone;
}

#include "path_pattern_ctx.h"
#include "../util/arr.h"
#include "ebnf_construction.h"
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

PathPatternCtx* PathPatternCtx_New() {
    PathPatternCtx *ctx = rm_malloc(sizeof(PathPatternCtx));
    ctx->patterns = array_new(PathPattern *, 1);
    return ctx;
}

void PathPatternCtx_AddPathPattern(PathPatternCtx *ctx, PathPattern *pattern) {
    ctx->patterns = array_append(ctx->patterns, pattern);
}

PathPattern* PathPatternCtx_GetPathPattern(PathPatternCtx *ctx, const char* name) {
    return _PathPatternCtx_FindPathPattern(ctx->patterns, name);
}

PathPattern **PathPatternCtx_GetDependencies(PathPatternCtx *ctx, AlgebraicExpression *ebnf_expr) {
    PathPattern **visited = array_new(PathPattern*,1);
    _PathPatternCtx_DFS(ctx, ebnf_expr, &visited);
    return visited;
}

PathPatternCtx *BuildPathPatternCtx(AST *ast, size_t required_dim) {
    PathPatternCtx *pathPatternCtx = PathPatternCtx_New();

    // TODO: CYPHER_AST_NAMED_PATH -> CYPHER_AST_NAMED_PATH_PATTERN
    const cypher_astnode_t **named_path_clauses = AST_GetClauses(ast, CYPHER_AST_NAMED_PATH);
    if (named_path_clauses) {
        uint named_path_count = array_len(named_path_clauses);
        for (int i = 0; i < named_path_count; ++i) {
            const cypher_astnode_t *identifier = cypher_ast_named_path_get_identifier(named_path_clauses[i]);
            const cypher_astnode_t *pattern_path_node = cypher_ast_named_path_get_path(named_path_clauses[i]);

            // Support global named patterns only like ()-/ ... /-()
            const cypher_astnode_t *path_pattern_node = cypher_ast_pattern_path_get_element(pattern_path_node, 1);

            const char *name = cypher_ast_identifier_get_name(identifier);
            EBNFBase *ebnf = BuildEBNFBaseFromPathPattern(path_pattern_node);

            PathPattern *path_pattern = PathPattern_New(name, ebnf, required_dim);
            PathPatternCtx_AddPathPattern(pathPatternCtx, path_pattern);
        }
    }
    return pathPatternCtx;
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

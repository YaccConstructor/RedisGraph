#include "path_pattern_ctx_construction.h"
#include "ebnf_construction.h"
#include "../util/arr.h"

PathPatternCtx *PathPatternCtx_Build(AST *ast, size_t required_dim) {
	PathPatternCtx *pathPatternCtx = PathPatternCtx_New(required_dim);
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
			EBNFBase *ebnf = EBNFBase_Build(path_pattern_node, pathPatternCtx);
			PathPattern *path_pattern = PathPattern_New(name, ebnf, required_dim);

			PathPatternCtx_AddPathPattern(pathPatternCtx, path_pattern);
		}
		array_free(named_path_clauses);
	}
	return pathPatternCtx;
}

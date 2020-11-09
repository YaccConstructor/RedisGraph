#include "path_pattern_ctx_construction.h"
#include "ebnf_construction.h"
#include "../util/arr.h"
#include "../arithmetic/algebraic_expression/utils.h"

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

			EBNFBase *ebnf_root = EBNFBase_Build(path_pattern_node, pathPatternCtx);
			PathPattern *path_pattern = PathPattern_New(name, ebnf_root, required_dim, false);

			EBNFBase *ebnf_root_transposed = EBNFGroup_New(CYPHER_REL_INBOUND, EBNF_ONE);
			EBNFBase_AddChild(ebnf_root_transposed, EBNFBase_Clone(ebnf_root));

			PathPattern *path_pattern_transposed = PathPattern_New(name, ebnf_root_transposed, required_dim, true);

			PathPatternCtx_AddPathPattern(pathPatternCtx, path_pattern);
			PathPatternCtx_AddPathPattern(pathPatternCtx, path_pattern_transposed);
		}
		array_free(named_path_clauses);
	}
	for (int i = 0; i < array_len(pathPatternCtx->patterns); ++i) {
		AlgebraicExpression_PopulateReferences(pathPatternCtx->patterns[i]->ae, pathPatternCtx);
	}
	return pathPatternCtx;
}

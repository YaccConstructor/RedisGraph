#include "ebnf_construction.h"

EBNFBase *_BuildEBNFBase(const cypher_astnode_t *node) {
    if (cypher_astnode_instanceof(node, CYPHER_AST_PATH_PATTERN_EXPRESSION)) {
        EBNFBase *seq = EBNFSequence_New();
        unsigned int nelem = cypher_ast_path_pattern_expression_get_nelements(node);
        for (int i = 0; i < nelem; ++i) {
            const cypher_astnode_t *child_node = cypher_ast_path_pattern_expression_get_element(node, i);
            EBNFBase *child = _BuildEBNFBase(child_node);
            EBNFBase_AddChild(seq, child);
        }
        return seq;
    } else if (cypher_astnode_instanceof(node, CYPHER_AST_PATH_PATTERN_ALTERNATIVE)) {
        EBNFBase *alt = EBNFAlternative_New();
        unsigned int nelem = cypher_ast_path_pattern_alternative_get_nelements(node);
        for (int i = 0; i < nelem; ++i) {
            const cypher_astnode_t *child_node = cypher_ast_path_pattern_alternative_get_element(node, i);
            EBNFBase *child = _BuildEBNFBase(child_node);
            EBNFBase_AddChild(alt, child);
        }
        return alt;
    } else if (cypher_astnode_instanceof(node, CYPHER_AST_PATH_PATTERN_BASE)) {
        EBNFBase *group = EBNFGroup_New(cypher_ast_path_pattern_base_get_direction(node), EBNF_ONE);
        EBNFBase *child = _BuildEBNFBase(cypher_ast_path_pattern_base_get_child(node));
        EBNFBase_AddChild(group, child);
        return group;
    } else if (cypher_astnode_instanceof(node, CYPHER_AST_PATH_PATTERN_EDGE)){
        const cypher_astnode_t *reltype = cypher_ast_path_pattern_edge_get_reltype(node);
        return EBNFEdge_New(cypher_ast_reltype_get_name(reltype));
    } else {
        assert(false && "EBNF TRANSLATION NOT IMPLEMETED");
    }
}

EBNFBase *BuildEBNFBaseFromPathPattern(const cypher_astnode_t *path_pattern) {
    REQUIRE_TYPE(path_pattern, CYPHER_AST_PATH_PATTERN, NULL);

    const cypher_astnode_t *expr = cypher_ast_path_pattern_get_expression(path_pattern);
    return _BuildEBNFBase(expr);
}

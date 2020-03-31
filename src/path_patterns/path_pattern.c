#include "path_pattern.h"
#include "../util/rmalloc.h"

PathPattern *PathPattern_New(const char *name, EBNFBase *ebnf, size_t reqiured_mdim) {
    PathPattern *pattern = rm_malloc(sizeof(PathPattern));

    pattern->name = name;
    pattern->ebnf_root = EBNFBase_Clone(ebnf);
    pattern->ae = NULL;
    GrB_Matrix_new(&pattern->m, GrB_BOOL, reqiured_mdim, reqiured_mdim);

    return pattern;
}

char *PathPattern_ToString(PathPattern *pattern) {
    char *buf = rm_malloc(sizeof(char) * 1024);
    sprintf(buf, "%s = %s\n\t %s", pattern->name, EBNFBase_ToStr(pattern->ebnf_root),
            AlgebraicExpression_ToStringDebug(pattern->ae));
    return buf;
}

void PathPattern_Free(PathPattern *pattern) {
    if (pattern->ebnf_root != NULL) {
        EBNFBase_Free(pattern->ebnf_root);
        pattern->ebnf_root = NULL;
    }
    if (pattern->ae != NULL) {
        AlgebraicExpression_Free(pattern->ae);
        pattern->ae = NULL;
    }
    if (pattern->m != GrB_NULL) {
        GrB_Matrix_free(&pattern->m);
        pattern->m = GrB_NULL;
    }
    rm_free(pattern);
}

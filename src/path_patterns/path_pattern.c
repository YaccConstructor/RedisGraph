#include "path_pattern.h"
#include "../util/rmalloc.h"
#include "../arithmetic/algebraic_expression/algebraic_expression_eval_dev.h"

PathPattern *PathPattern_New(const char *name, EBNFBase *ebnf, size_t reqiured_mdim) {
    PathPattern *pattern = rm_malloc(sizeof(PathPattern));

	GrB_Matrix_new(&pattern->m, GrB_BOOL, reqiured_mdim, reqiured_mdim);
	GrB_Matrix_new(&pattern->src, GrB_BOOL, reqiured_mdim, reqiured_mdim);

    pattern->name = name;
    pattern->ebnf_root = ebnf;

    AlgebraicExpression *ae = AlgebraicExpression_FromEbnf(ebnf);
    AlgebraicExpression_MultiplyToTheLeft(&ae, pattern->src);
	AlgebraicExpression_Optimize(&ae);

//	printf("total show new\n");
//	AlgebraicExpression_TotalShow(ae);

	pattern->ae = ae;

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
    if (pattern->src != GrB_NULL) {
    	GrB_Matrix_free(&pattern->src);
    	pattern->src = GrB_NULL;
    }
    rm_free(pattern);
}

PathPattern *PathPattern_Clone(PathPattern *other) {
	PathPattern *clone = rm_malloc(sizeof(PathPattern));

	clone->name = other->name;
	clone->ebnf_root = EBNFBase_Clone(other->ebnf_root);

	GrB_Matrix_dup(&clone->m, other->m);
	GrB_Matrix_dup(&clone->src, other->src);

	AlgebraicExpression *ae = AlgebraicExpression_FromEbnf(clone->ebnf_root);
	AlgebraicExpression_MultiplyToTheLeft(&ae, clone->src);
	AlgebraicExpression_Optimize(&ae);

	clone->ae = ae;
    return clone;
}

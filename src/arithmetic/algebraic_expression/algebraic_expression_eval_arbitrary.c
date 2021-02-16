/*
* Copyright 2018-2020 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "utils.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"
#include "../../path_patterns/path_pattern_ctx.h"

// Forward declarations
GrB_Matrix _AlgebraicExpression_EvalArbitrary(const AlgebraicExpression *exp);

static GrB_Matrix _Eval_Operand( const AlgebraicExpression *exp) {
    assert(exp);
    GrB_Matrix res = GrB_NULL;
    if (GrB_Matrix_dup(&res, exp->operand.matrix) != GrB_SUCCESS) {
        const char *error_msg = NULL;
		GrB_error(&error_msg, res);
		fprintf(stderr, "%s", error_msg);
		ASSERT(false);
    }
    return res;
}

static GrB_Matrix _Eval_TransposeArbitrary( const AlgebraicExpression *exp) {
    // In path patterns transpose operation can contain another operation.
    // This function is called only if transpose is the root of algebraic
    // expression. Otherwise transpose is evaluated via descriptor.

    assert(exp && AlgebraicExpression_ChildCount(exp) == 1);

    GrB_Info info;
    GrB_Matrix res = GrB_NULL;
    GrB_Index nrows;                // Number of rows of operand.
    GrB_Index ncols;                // Number of columns of operand.

    AlgebraicExpression *child = FIRST_CHILD(exp);

    if (child->type == AL_OPERAND) {
        GrB_Matrix_nrows(&nrows, child->operand.matrix);
        GrB_Matrix_ncols(&ncols, child->operand.matrix);
        info = GrB_Matrix_new(&res, GrB_BOOL, nrows, ncols);
        if(info != GrB_SUCCESS) {
            const char *error_msg = NULL;
		    GrB_error(&error_msg, res);
		    fprintf(stderr, "%s", error_msg);
		    ASSERT(false);
        }

        if (GrB_transpose(res, GrB_NULL, GrB_NULL, child->operand.matrix, GrB_NULL) != GrB_SUCCESS) {
            const char *error_msg = NULL;
		    GrB_error(&error_msg, res);
		    fprintf(stderr, "%s", error_msg);
		    ASSERT(false);
        }
    } else {
        res = _AlgebraicExpression_EvalArbitrary(child);
        GrB_transpose(res, GrB_NULL, GrB_NULL, res, GrB_NULL);
    }
    return res;
}

static GrB_Matrix _Eval_AddArbitrary(const AlgebraicExpression *exp) {
    assert(exp && AlgebraicExpression_ChildCount(exp) > 1);

    GrB_Matrix A = GrB_NULL;        // Left operand.
    GrB_Matrix B = GrB_NULL;        // Right operand.
    GrB_Matrix res = GrB_NULL;     // Intermidate matrix.

    bool need_free_A = false;
    bool need_free_B = false;

    GrB_Info info;
    GrB_Index nrows;                // Number of rows of operand.
    GrB_Index ncols;                // Number of columns of operand.
    GrB_Descriptor desc = GrB_NULL; // Descriptor used for transposing operands.

    GrB_Descriptor_new(&desc);

    // Get left and right operands.
    AlgebraicExpression *left = CHILD_AT(exp, 0);
    AlgebraicExpression *right = CHILD_AT(exp, 1);

    /* If left operand is a matrix or transpose operation with one operand matrix, simply get it.
     * Otherwise evaluate left hand or child of transpose operation.
     * In this case we need to free A. */
    if(left->type == AL_OPERAND) {
        A = left->operand.matrix;
    } else {
        if(left->operation.op == AL_EXP_TRANSPOSE) {
            GrB_Descriptor_set(desc, GrB_INP0, GrB_TRAN);
            AlgebraicExpression *child = CHILD_AT(left, 0);

            if (child->type == AL_OPERAND) {
                A = child->operand.matrix;
            } else {
                A = _AlgebraicExpression_EvalArbitrary(child);
                need_free_A = true;
            }
        } else {
            A = _AlgebraicExpression_EvalArbitrary(left);
            need_free_A = true;
        }
    }

	/* If left operand is a matrix or transpose operation with one operand matrix, simply get it.
	 * Otherwise evaluate left hand or child of transpose operation.
 	 * In this case we need to free B. */
    if(right->type == AL_OPERAND) {
        B = right->operand.matrix;
    } else {
        if(right->operation.op == AL_EXP_TRANSPOSE) {
            GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
            AlgebraicExpression *child = CHILD_AT(right, 0);

            if (child->type == AL_OPERAND) {
                B = child->operand.matrix;
            } else {
                B = _AlgebraicExpression_EvalArbitrary(child);
                need_free_B = true;
            }
        } else {
            B = _AlgebraicExpression_EvalArbitrary(right);
            need_free_B = true;
        }
    }

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    info = GrB_Matrix_new(&res, GrB_BOOL, nrows, ncols);
    if(info != GrB_SUCCESS) {
        const char *error_msg = NULL;
		GrB_error(&error_msg, res);
		fprintf(stderr, "%s", error_msg);
		ASSERT(false);
    }

    // Perform addition.
    if(GrB_Matrix_eWiseAdd_Semiring(res, GrB_NULL, GrB_NULL, GxB_ANY_PAIR_BOOL, A, B,
                                    desc) != GrB_SUCCESS) {
        const char *error_msg = NULL;
		GrB_error(&error_msg, res);
		fprintf(stderr, "%s", error_msg);
		ASSERT(false);
    }

    // Reset descriptor and free matrices
    GrB_Descriptor_set(desc, GrB_INP0, GxB_DEFAULT);

    if (need_free_A) {
        GrB_Matrix_free(&A);
        need_free_A = false;
    }
    if (need_free_B) {
        GrB_Matrix_free(&B);
        need_free_B = false;
    }

    uint child_count = AlgebraicExpression_ChildCount(exp);
    // Expression has more than 2 operands, e.g. A+B+C...
    for(uint i = 2; i < child_count; i++) {
        // Reset descriptor.
        GrB_Descriptor_set(desc, GrB_INP1, GxB_DEFAULT);
        right = CHILD_AT(exp, i);

        if(right->type == AL_OPERAND) {
            B = right->operand.matrix;
        } else {
            if(right->operation.op == AL_EXP_TRANSPOSE) {
                GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
                AlgebraicExpression *child = CHILD_AT(right, 0);

                if (child->type == AL_OPERAND) {
                    B = child->operand.matrix;
                } else {
                    B = _AlgebraicExpression_EvalArbitrary(child);
                    need_free_B = true;
                }
            } else {
                // Evaluate
                B = _AlgebraicExpression_EvalArbitrary(right);
                need_free_B = true;
            }
        }

        // Perform addition.
        if(GrB_Matrix_eWiseAdd_Semiring(res, GrB_NULL, GrB_NULL, GxB_ANY_PAIR_BOOL, res, B,
                                        desc) != GrB_SUCCESS) {
            const char *error_msg = NULL;
		    GrB_error(&error_msg, res);
		    fprintf(stderr, "%s", error_msg);
		    ASSERT(false);
        }

        if (need_free_B) {
            GrB_Matrix_free(&B);
            need_free_B = false;
        }
    }

    GrB_free(&desc);
    return res;
}

static GrB_Matrix _Eval_MulArbitrary(const AlgebraicExpression *exp) {
    assert(exp && AlgebraicExpression_ChildCount(exp) > 1);

    GrB_Matrix A = GrB_NULL;        // Left operand.
    GrB_Matrix B = GrB_NULL;        // Right operand.
    GrB_Matrix res = GrB_NULL;    // Intermidate matrix.

    bool need_free_A = false;
    bool need_free_B = false;

    GrB_Info info;
    GrB_Index nrows;                // Number of rows of operand.
    GrB_Index ncols;                // Number of columns of operand.
    GrB_Descriptor desc = GrB_NULL; // Descriptor used for transposing operands.

    GrB_Descriptor_new(&desc);

    // Get left and right operands.
    AlgebraicExpression *left = CHILD_AT(exp, 0);
    AlgebraicExpression *right = CHILD_AT(exp, 1);

	/* If left operand is a matrix or transpose operation with one operand matrix, simply get it.
	 * Otherwise evaluate left hand or child of transpose operation.
	 * In this case we need to free A. */
    if(left->type == AL_OPERAND) {
        A = left->operand.matrix;
    } else {
        if(left->operation.op == AL_EXP_TRANSPOSE) {
            GrB_Descriptor_set(desc, GrB_INP0, GrB_TRAN);
            AlgebraicExpression *child = CHILD_AT(left, 0);

            if (child->type == AL_OPERAND) {
                A = child->operand.matrix;
            } else {
                A = _AlgebraicExpression_EvalArbitrary(child);
                need_free_A = true;
            }
        } else {
            // Evaluate
            A = _AlgebraicExpression_EvalArbitrary(left);
            need_free_A = true;
        }
    }

	/* If left operand is a matrix or transpose operation with one operand matrix, simply get it.
	 * Otherwise evaluate left hand or child of transpose operation.
	 * In this case we need to free B. */
    if(right->type == AL_OPERAND) {
        B = right->operand.matrix;
    } else {
        if(right->operation.op == AL_EXP_TRANSPOSE) {
            GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
            AlgebraicExpression *child = CHILD_AT(right, 0);

            if (child->type == AL_OPERAND) {
                B = child->operand.matrix;
            } else {
                B = _AlgebraicExpression_EvalArbitrary(child);
                need_free_B = true;
            }
        } else {
            // Evaluate
            B = _AlgebraicExpression_EvalArbitrary(right);
            need_free_B = true;
        }
    }

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    info = GrB_Matrix_new(&res, GrB_BOOL, nrows, ncols);
    if(info != GrB_SUCCESS) {
        const char *error_msg = NULL;
		GrB_error(&error_msg, res);
		fprintf(stderr, "%s", error_msg);
		ASSERT(false);
    }

    // Perform addition.
    if (GrB_mxm(res, GrB_NULL, GrB_NULL, GxB_ANY_PAIR_BOOL, A, B, desc) != GrB_SUCCESS) {
        const char *error_msg = NULL;
        GrB_error(&error_msg, res);
		fprintf(stderr, "%s", error_msg);
		ASSERT(false);
    }

    // Reset descriptor and free matrices
    GrB_Descriptor_set(desc, GrB_INP0, GxB_DEFAULT);

    if (need_free_A) {
        GrB_Matrix_free(&A);
        need_free_A = false;
    }
    if (need_free_B) {
        GrB_Matrix_free(&B);
        need_free_B = false;
    }

    uint child_count = AlgebraicExpression_ChildCount(exp);
    // Expression has more than 2 operands, e.g. A+B+C...
    for(uint i = 2; i < child_count; i++) {
        // Reset descriptor.
        GrB_Descriptor_set(desc, GrB_INP1, GxB_DEFAULT);
        right = CHILD_AT(exp, i);

        if(right->type == AL_OPERAND) {
            B = right->operand.matrix;
        } else {
            if(right->operation.op == AL_EXP_TRANSPOSE) {
                GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
                AlgebraicExpression *child = CHILD_AT(right, 0);

                if (child->type == AL_OPERAND) {
                    B = child->operand.matrix;
                } else {
                    B = _AlgebraicExpression_EvalArbitrary(child);
                    need_free_B = true;
                }
            } else {
                // Evaluate
                B = _AlgebraicExpression_EvalArbitrary(right);
                need_free_B = true;
            }
        }

        // Perform addition.
        if (GrB_mxm(res, GrB_NULL, GrB_NULL, GxB_ANY_PAIR_BOOL, res, B, desc) != GrB_SUCCESS) {
            const char *error_msg = NULL;
		    GrB_error(&error_msg, res);
		    fprintf(stderr, "%s", error_msg);
		    ASSERT(false);
        }

        if (need_free_B) {
            GrB_Matrix_free(&B);
            need_free_B = false;
        }
    }

    GrB_free(&desc);
    return res;
}


GrB_Matrix _AlgebraicExpression_EvalArbitrary(const AlgebraicExpression *exp) {
    // Return a NEW matrix with computing result
    assert(exp);

    // Perform operation.
    switch (exp->type) {
        case AL_OPERATION:
            switch (exp->operation.op) {
                case AL_EXP_MUL:
                    return _Eval_MulArbitrary(exp);

                case AL_EXP_ADD:
                    return _Eval_AddArbitrary(exp);

                case AL_EXP_TRANSPOSE:
                    return _Eval_TransposeArbitrary(exp);

                default:
                    assert("Unknown algebraic expression operation" && false);
            }
            break;
        case AL_OPERAND:
            return _Eval_Operand(exp);
        default:
            assert(false && "Unknown algebraic expression node type");
            break;
    }
}

GrB_Matrix AlgebraicExpression_EvalArbitrary(const AlgebraicExpression *exp) {
    // Fetch operands and evalute operation.
    // Return new matrix with computing result.
    // In case of operand return duplicate.
    assert(exp);

    _AlgebraicExpression_PopulateOperands((AlgebraicExpression *)exp, QueryCtx_GetGraphCtx());

    return _AlgebraicExpression_EvalArbitrary(exp);
}

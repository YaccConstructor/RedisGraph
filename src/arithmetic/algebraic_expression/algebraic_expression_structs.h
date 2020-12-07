#pragma once

#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

static GrB_Matrix IDENTITY_MATRIX = (GrB_Matrix)0x31032017;  // Identity matrix.

// Matrix, vector operations.
typedef enum {
	AL_EXP_ADD = 1,                 // Matrix addition.
	AL_EXP_MUL = (1 << 1),          // Matrix multiplication.
	AL_EXP_POW = (1 << 2),          // Matrix raised to a power.
	AL_EXP_TRANSPOSE = (1 << 3),    // Matrix transpose.
} AL_EXP_OP;

#define AL_EXP_ALL (AL_EXP_ADD | AL_EXP_MUL | AL_EXP_POW | AL_EXP_TRANSPOSE)

// Type of node within an algebraic expression
typedef enum {
	AL_OPERAND = 1,
	AL_OPERATION  = (1 << 1),
} AlgebraicExpressionType;

typedef struct {
	char *name;
	bool transposed;
} AlgExpReference;

/* Forward declarations. */
typedef struct AlgebraicExpression AlgebraicExpression;

struct AlgebraicExpression {
	union {
		struct {
			bool diagonal;          // Diagonal matrix.
			bool bfree;             // If the matrix is scoped to this expression, it should be freed with it.
			GrB_Matrix matrix;      // Matrix operand.
			const char *src;        // Alias given to operand's rows (src node).
			const char *dest;       // Alias given to operand's columns (destination node).
			const char *edge;       // Alias given to operand (edge).
			const char *label;      // Label attached to matrix.
			AlgExpReference reference;
		} operand;
		struct {
			AL_EXP_OP op;                   // Operation: `*`,`+`,`transpose`
			AlgebraicExpression **children; // Child nodes.
		} operation;
	};
	AlgebraicExpressionType type;   // Type of node, either an operation or an operand.
};
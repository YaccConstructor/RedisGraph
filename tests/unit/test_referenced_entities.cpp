/*
* Copyright 2018-2019 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "gtest.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "../../src/util/rmalloc.h"
#include "../../src/util/arr.h"
#include "../../src/ast/ast.h"
#include "../../src/parser/parser.h"

#ifdef __cplusplus
}
#endif

class TestReferencedEntities: public ::testing::Test {
  protected:
	static void SetUpTestCase() {// Use the malloc family for allocations
		Alloc_Reset();
	}
};

// Parse query to AST.
AST *buildAST(const char *query) {
	cypher_parse_result_t *parse_result = parse(query);
	return AST_Build(parse_result);
}

uint *getASTSegmentIndices(AST *ast) {
	// Retrieve the indices of each WITH clause to properly set the bounds of each segment.
	uint *segment_indices = AST_GetClauseIndices(ast, CYPHER_AST_WITH);
	bool query_has_return = AST_ContainsClause(ast, CYPHER_AST_RETURN);
	if(query_has_return) {
		segment_indices = array_append(segment_indices, cypher_ast_query_nclauses(ast->root) - 1);
	}
	segment_indices = array_append(segment_indices, cypher_ast_query_nclauses(ast->root));
	return segment_indices;
}

TEST_F(TestReferencedEntities, TestMatch) {
	// Test for no inline filters.
	char *q = "MATCH (n)";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(0, raxSize(astSegment->referenced_entities));

	q = "MATCH (n)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(0, raxSize(astSegment->referenced_entities));

	// Inline node label filter.
	q = "MATCH (n:X)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline node property filter.
	q = "MATCH (n {val:42})-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline edge relationship type.
	q = "MATCH (n)-[e:T]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline edge properties.
	q = "MATCH (n)-[e {val:42}]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Explicit filters: EQ, NE, GT, LT, IS NULL, IS NOT NULL.
	q = "MATCH (n)-[e]->(m) WHERE n.v = 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	q = "MATCH (n)-[e]->(m) WHERE n.v <> 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	q = "MATCH (n)-[e]->(m) WHERE n.v > 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	q = "MATCH (n)-[e]->(m) WHERE n.v < 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	q = "MATCH (n)-[e]->(m) WHERE n.v IS NULL 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	q = "MATCH (n)-[e]->(m) WHERE n.v IS NOT NULL 1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

}

TEST_F(TestReferencedEntities, TestSet) {
	// Simple match and set property.
	char *q = "MATCH (n) SET n.v=1";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));

	// Path match and set property.
	q = "MATCH (n)-[e]->(m) SET n.v=1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Reference from MATCH and SET caluse.
	q = "MATCH (n)-[e:r]->(m) SET n.v=1";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(2, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
}

TEST_F(TestReferencedEntities, TestMerge) {
	// Simple merge.
	char *q = "MERGE (n)";
	AST *ast = buildAST(q);
	uint *segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));

	AST *astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(0, raxSize(astSegment->referenced_entities));

	q = "MERGE (n)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(0, raxSize(astSegment->referenced_entities));

	// Inline node label filter.
	q = "MERGE (n:X)-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline node property filter.
	q = "MERGE (n {val:42})-[e]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline edge relationship type.
	q = "MERGE (n)-[e:T]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Inline edge properties.
	q = "MERGE (n)-[e {val:42}]->(m)";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(1, raxSize(astSegment->referenced_entities));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));


	// On match.
	q = "MERGE (n)-[e]->(m) ON MATCH SET n.v=1, m.v=2";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(2, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// On create.
	q = "MERGE (n)-[e]->(m) ON CREATE SET n.v=1, m.v=2";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(2, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_EQ(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));

	// Mixed.
	q = "MERGE (n)-[e]->(m) ON CREATE SET n.v=1, m.v=2 ON MATCH SET e.v = 3";
	ast = buildAST(q);
	segmentIndices = getASTSegmentIndices(ast);
	ASSERT_EQ(1, array_len(segmentIndices));
	astSegment = AST_NewSegment(ast, 0, segmentIndices[0]);
	ASSERT_EQ(3, raxSize(astSegment->referenced_entities));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"n", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"e", 1));
	ASSERT_NE(raxNotFound, raxFind(astSegment->referenced_entities, (unsigned char *)"m", 1));
}

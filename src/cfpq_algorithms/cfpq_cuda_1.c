#include "cfpq_algorithms.h"
#include <algorithms/cuda/m4ri/include/m4ri.h>

int CFPQ_gpu1(RedisModuleCtx *ctx, GraphContext* gc, Grammar* grammar, CfpqResponse* response) {
  size_t graph_size = Graph_RequiredMatrixDim(gc->g);
  size_t relations_count = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
  GrB_Matrix relations[relations_count];
  const char* relations_names[relations_count];

  for (size_t i = 0; i < relations_count; i++) {
    relations_names[i] = gc->relation_schemas[i]->name;
    relations[i] = gc->g->relations[i];
  }

  //TODO check result code
  m4ri(grammar, response, relations, relations_names,
                relations_count, graph_size);


  return REDISMODULE_OK;
}

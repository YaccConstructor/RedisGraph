#pragma once

#include "path_pattern_ctx.h"
#include "../ast/ast.h"

PathPatternCtx *PathPatternCtx_Build(AST *ast, size_t required_dim);

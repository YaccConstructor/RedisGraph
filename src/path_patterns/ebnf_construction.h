#pragma once

#include "ebnf.h"
#include <astnode.h>

EBNFBase *BuildEBNFBaseFromPathPattern(const cypher_astnode_t *path_pattern);
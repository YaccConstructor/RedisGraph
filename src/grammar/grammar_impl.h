#pragma once

#include <stdint.h>
#include "conf.h"

typedef struct {
    RuleItem count;
    char items[];
} FixedDict;

typedef struct {
    RuleItem count;
    char items[3 * MAX_GRAMMAR_SIZE];
} NontermDict;

typedef struct {
    RuleItem count;
    char items[MAX_GRAMMAR_SIZE];
} TokenDict;

void FixedDict_Init(FixedDict *dict);
RuleItem FixedDict_InsertOrGet(FixedDict *dict, char token);

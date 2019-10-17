#pragma once

#include <stdio.h>
#include "conf.h"

#define GRAMMAR_LOAD_ERROR 0
#define GRAMMAR_LOAD_SUCCESS 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    MapperIndex count;
    char items[MAX_NONTERM_COUNT][MAX_ITEM_NAME_LEN];
} NontermMapper;

typedef struct {
    MapperIndex count;
    char items[MAX_GRAMMAR_SIZE][MAX_ITEM_NAME_LEN];
} TokenMapper;

typedef struct {
    MapperIndex l;
    MapperIndex r1;
    MapperIndex r2;
} ComplexRule;

typedef struct {
    MapperIndex l;
    MapperIndex r;
} SimpleRule;

typedef struct {
    ComplexRule complex_rules[MAX_GRAMMAR_SIZE];
    int complex_rules_count;

    SimpleRule simple_rules[MAX_GRAMMAR_SIZE];
    int simple_rules_count;

    NontermMapper nontermMapper;
    TokenMapper tokenMapper;
} Grammar;

int Grammar_Load(Grammar *gr, FILE *f);
void Grammar_Init(Grammar *gr);

void Grammar_AddSimpleRule(Grammar *gr, MapperIndex l, MapperIndex r);
void Grammar_AddComplexRule(Grammar *gr, MapperIndex l, MapperIndex r1, MapperIndex r2);

#ifdef __cplusplus
}
#endif

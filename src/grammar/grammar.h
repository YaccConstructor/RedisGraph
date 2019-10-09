#pragma once

#include <bits/types/FILE.h>
#include "conf.h"

typedef struct {
    MapperIndex count;
    char items[3 * MAX_GRAMMAR_SIZE];
} NontermMapper;

typedef struct {
    MapperIndex count;
    char items[MAX_GRAMMAR_SIZE];
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

void Grammar_Load(Grammar *gr, FILE *f);
void Grammar_Init(Grammar *gr);

void Grammar_AddSimpleRule(Grammar *gr, MapperIndex l, MapperIndex r);
void Grammar_AddComplexRule(Grammar *gr, MapperIndex l, MapperIndex r1, MapperIndex r2);

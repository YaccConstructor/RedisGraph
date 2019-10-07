#pragma once

#include <bits/types/FILE.h>
#include "conf.h"

typedef struct {
    RuleItem l;
    RuleItem r1;
    RuleItem r2;
} ComplexRule;

typedef struct {
    RuleItem l;
    RuleItem r;
} SimpleRule;

typedef struct {
    ComplexRule complex_rules[MAX_GRAMMAR_SIZE];
    int complex_rules_count;

    SimpleRule simple_rules[MAX_GRAMMAR_SIZE];
    int simple_rules_count;
} Grammar;

void Grammar_Load(Grammar *gr, FILE *f);
void Grammar_Init(Grammar *gr);
void Grammar_AddSimpleRule(Grammar *gr, RuleItem l, RuleItem r);
void Grammar_AddComplexRule(Grammar *gr, RuleItem l, RuleItem r1, RuleItem r2);

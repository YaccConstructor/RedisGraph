#include <stdio.h>
#include <assert.h>
#include "grammar.h"
#include "grammar_impl.h"
#include "helpers.h"

void Grammar_Load(Grammar *gr, FILE *f) {
    Grammar_Init(gr);

    NontermDict nontermDict;
    FixedDict_Init((FixedDict *) &nontermDict);

    TokenDict tokenDict;
    FixedDict_Init((FixedDict *) &tokenDict);

    char *grammar_buf;
    size_t buf_size = 0;

    while (getline(&grammar_buf, &buf_size, f) != -1) {
        str_strip(grammar_buf);

        char l, r1, r2;
        int nitems = sscanf(grammar_buf, "%c %c %c", &l, &r1, &r2);

        if (nitems == 2) {
            int gr_l = FixedDict_InsertOrGet((FixedDict *) &nontermDict, l);
            int gr_r = FixedDict_InsertOrGet((FixedDict *) &tokenDict, r1);

            Grammar_AddSimpleRule(gr, gr_l, gr_r);
        } else if (nitems == 3) {
            int gr_l = FixedDict_InsertOrGet((FixedDict *) &nontermDict, l);
            int gr_r1 = FixedDict_InsertOrGet((FixedDict *) &nontermDict, r1);
            int gr_r2 = FixedDict_InsertOrGet((FixedDict *) &nontermDict, r2);

            Grammar_AddComplexRule(gr, gr_l, gr_r1, gr_r2);
        }
    }
}

void Grammar_Init(Grammar *gr) {
    gr->complex_rules_count = 0;
    gr->simple_rules_count = 0;
}

void Grammar_AddSimpleRule(Grammar *gr, RuleItem l, RuleItem r) {
    // TODO: replace assert to something else
    assert(gr->simple_rules_count != MAX_GRAMMAR_SIZE);

    SimpleRule newSimpleRule = {.l = l, .r = r};
    gr->simple_rules[gr->simple_rules_count] = newSimpleRule;
    gr->simple_rules_count++;
}

void Grammar_AddComplexRule(Grammar *gr, RuleItem l, RuleItem r1, RuleItem r2) {
    // TODO: replace assert to something else
    assert(gr->complex_rules_count != MAX_GRAMMAR_SIZE);

    ComplexRule newComplexRule = {.l = l, .r1 = r1, .r2 = r2};
    gr->complex_rules[gr->complex_rules_count] = newComplexRule;
    gr->complex_rules_count++;
}

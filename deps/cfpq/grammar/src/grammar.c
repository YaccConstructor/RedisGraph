#include <stdio.h>
#include <assert.h>
#include "grammar.h"
#include "item_mapper.h"
#include "helpers.h"

void Grammar_Init(Grammar *gr) {
    gr->complex_rules_count = 0;
    gr->simple_rules_count = 0;

    ItemMapper_Init((ItemMapper *) &gr->nontermMapper);
    ItemMapper_Init((ItemMapper *) &gr->tokenMapper);
}

int Grammar_Load(Grammar *gr, FILE *f) {
    Grammar_Init(gr);

    char *grammar_buf;
    size_t buf_size = 0;

    while (getline(&grammar_buf, &buf_size, f) != -1) {
        str_strip(grammar_buf);

        char l[MAX_ITEM_NAME_LEN], r1[MAX_ITEM_NAME_LEN], r2[MAX_ITEM_NAME_LEN];
        int nitems = sscanf(grammar_buf, "%s %s %s", l, r1, r2);

        if (nitems == 2) {
            int gr_l = ItemMapper_Insert((ItemMapper *) &gr->nontermMapper, l);
            int gr_r = ItemMapper_Insert((ItemMapper *) &gr->tokenMapper, r1);

            Grammar_AddSimpleRule(gr, gr_l, gr_r);
        } else if (nitems == 3) {
            int gr_l = ItemMapper_Insert((ItemMapper *) &gr->nontermMapper, l);
            int gr_r1 = ItemMapper_Insert((ItemMapper *) &gr->nontermMapper, r1);
            int gr_r2 = ItemMapper_Insert((ItemMapper *) &gr->nontermMapper, r2);

            Grammar_AddComplexRule(gr, gr_l, gr_r1, gr_r2);
        } else {
            return GRAMMAR_LOAD_ERROR;
        }
    }
    return GRAMMAR_LOAD_SUCCESS;
}

void Grammar_AddSimpleRule(Grammar *gr, MapperIndex l, MapperIndex r) {
    // TODO: replace assert to something else
    assert(gr->simple_rules_count != MAX_GRAMMAR_SIZE);

    SimpleRule newSimpleRule = {.l = l, .r = r};
    gr->simple_rules[gr->simple_rules_count] = newSimpleRule;
    gr->simple_rules_count++;
}

void Grammar_AddComplexRule(Grammar *gr, MapperIndex l, MapperIndex r1, MapperIndex r2) {
    // TODO: replace assert to something else
    assert(gr->complex_rules_count != MAX_GRAMMAR_SIZE);

    ComplexRule newComplexRule = {.l = l, .r1 = r1, .r2 = r2};
    gr->complex_rules[gr->complex_rules_count] = newComplexRule;
    gr->complex_rules_count++;
}

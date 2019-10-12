#include "../grammar.h"
#include "../item_mapper.h"
#include <stdio.h>

int main() {
    // Small example of loading and out grammar.

    FILE *f = fopen("toy_cfg.txt", "r");
    Grammar gr;
    Grammar_Load(&gr, f);

    for (int i = 0; i < gr.simple_rules_count; i++) {
        printf("%s -> %s\n", ItemMapper_Map((ItemMapper *) &gr.nontermMapper, gr.simple_rules[i].l),
                                    ItemMapper_Map((ItemMapper *) &gr.tokenMapper, gr.simple_rules[i].r));
    }
    for (int i = 0; i < gr.complex_rules_count; ++i) {
        printf("%s -> %s %s\n", ItemMapper_Map((ItemMapper *) &gr.nontermMapper, gr.complex_rules[i].l),
               ItemMapper_Map((ItemMapper *) &gr.nontermMapper, gr.complex_rules[i].r1),
               ItemMapper_Map((ItemMapper *) &gr.nontermMapper, gr.complex_rules[i].r2));
    }
    return 0;
}
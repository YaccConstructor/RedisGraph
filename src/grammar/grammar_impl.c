#include "grammar_impl.h"

void FixedDict_Init(FixedDict *dict) {
    dict->count = 0;
}

RuleItem FixedDict_InsertOrGet(FixedDict *dict, char token) {
    for (RuleItem i = 0; i < dict->count; i++) {
        if (dict->items[i] == token) {
            return i;
        }
    }
    dict->items[dict->count] = token;
    return dict->count++;
}

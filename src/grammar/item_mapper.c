#include <assert.h>

#include "item_mapper.h"
#include "string.h"

MapperIndex ItemMapper_GetPlaceIndex(ItemMapper *dict, const char *token) {
    for (MapperIndex i = 0; i < dict->count; i++) {
        if (strcmp(dict->items[i], token) ==  0) {
            return i;
        }
    }
    return dict->count;
}

void ItemMapper_Init(ItemMapper *dict) {
    dict->count = 0;
}

MapperIndex ItemMapper_Insert(ItemMapper *dict, const char* token) {
    MapperIndex i = ItemMapper_GetPlaceIndex(dict, token);
    if (i < dict->count) {
        return i;
    } else {
        strcpy(dict->items[dict->count], token);
        return dict->count++;
    }
}

int ItemMapper_Find(ItemMapper *dict, const char* token) {
    return ItemMapper_GetPlaceIndex(dict, token) == dict->count ? ITEM_NOT_EXIST : ITEM_EXIST;
}

char* ItemMapper_Map(ItemMapper *dict, MapperIndex mapperIdex) {
    assert(mapperIdex < dict->count);
    return dict->items[mapperIdex];
}

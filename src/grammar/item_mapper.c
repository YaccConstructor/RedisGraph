#include <assert.h>
#include "item_mapper.h"

MapperIndex _ItemMapper_GetPlaceIndex(ItemMapper *dict, char token) {
    for (MapperIndex i = 0; i < dict->count; i++) {
        if (dict->items[i] == token) {
            return i;
        }
    }
    return dict->count;
}

void ItemMapper_Init(ItemMapper *dict) {
    dict->count = 0;
}

MapperIndex ItemMapper_Insert(ItemMapper *dict, char token) {
    MapperIndex i = _ItemMapper_GetPlaceIndex(dict, token);
    if (i < dict->count) {
        return i;
    } else {
        dict->items[dict->count] = token;
        return dict->count++;
    }
}

IM_FindRes ItemMapper_Find(ItemMapper *dict, char token) {
    return _ItemMapper_GetPlaceIndex(dict, token) == dict->count ? ItemMapper_NOT_EXIST : ItemMapper_EXIST;
}

char ItemMapper_Map(ItemMapper *dict, char mapperIdex) {
    assert(mapperIdex < dict->count);
    return dict->items[mapperIdex];
}

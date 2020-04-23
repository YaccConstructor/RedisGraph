#include "Gvector.h"
#include <assert.h>

void vector_int_init(vector *v)
{
    v->size = 0;
    v->init = 1;
}

void vector_int_append(vector *v, int64_t data)
{
    assert(v->init == 1);
    assert(v->size < MAX_SIZE);
    v->items[v->size] = data;
    v->size++;
}

void vector_int_set_element(vector *v, int64_t newdata, unsigned int index)
{
    assert(v->init == 1);
    assert(v->size > index);

    v->items[index] = newdata;
}

int64_t vector_int_get_element_by_index(vector *v, unsigned int index)
{
    assert(v->init == 1);
    assert(v->size > index);

    return v->items[index];
}

unsigned int vector_int_get_size(vector *v)
{
    assert(v->init == 1);
    return v->size;
}

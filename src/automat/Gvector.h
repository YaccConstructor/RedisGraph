#pragma once

#include "GraphBLAS.h"

typedef struct
{
    GrB_Vector v;
    int init;
    unsigned int size;
} vector;

void vector_int_init(vector *v);
void vector_int_delete(vector *v);
void vector_int_append(vector *v, int32_t data);
void vector_int_set_element(vector *v, int32_t newdata, unsigned int index);
int vector_int_get_element_by_index(vector *v, unsigned int index);
void vector_int_resize(vector *v, unsigned int newsize);
unsigned int vector_int_get_size(vector *v);

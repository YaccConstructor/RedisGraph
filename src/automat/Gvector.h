#pragma once

#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

#define MAX_SIZE 100

typedef struct
{
    uint16_t size;
    int init;
    int64_t items[MAX_SIZE];
} vector;

void vector_int_init(vector *v);
void vector_int_append(vector *v, int64_t data);
void vector_int_set_element(vector *v, int64_t newdata, unsigned int index);
int64_t vector_int_get_element_by_index(vector *v, unsigned int index);
unsigned int vector_int_get_size(vector *v);

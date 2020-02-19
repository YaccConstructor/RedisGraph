#include "Gvector.h"
#include <assert.h>

void vector_int_init(vector *v)
{
    GrB_Vector_new(&v->v, GrB_INT32, 0);
    v->init = 1;
    v->size = 0;
}

void vector_int_delete(vector *v)
{
    assert(v->init == 1);
    GrB_Vector_free(&v->v);
    v->init = 0;
}

void vector_int_append(vector *v, int32_t data)
{
    assert(v->init == 1);

    GxB_Vector_resize(v->v, v->size + 1);
    v->size++;

    GrB_Info inf = GrB_Vector_setElement_INT64(v->v, data, v->size - 1);
    assert(inf == GrB_SUCCESS);
}

void vector_int_set_element(vector *v, int32_t newdata, unsigned int index)
{
    assert(v->init == 1);
    assert(index < v->size);

    GrB_Info info = GrB_Vector_setElement_INT32(v->v, newdata, index);
    assert(info == GrB_SUCCESS);
}

int32_t vector_int_get_element_by_index(vector *v, unsigned int index)
{
    assert(v->init == 1);
    assert(index < v->size);

    int32_t result = 0;
    GrB_Info info = GrB_Vector_extractElement_INT32(&result, v->v, index);
    assert(info == GrB_SUCCESS);

    return result;
}

//void vector_int_resize(vector *v, unsigned int newsize)
//{
//    assert(v->init == 1);

//    GrB_Vector new_vec;
//    GrB_Vector_new(&new_vec, GrB_INT32, v->size);
//    GrB_Vector_dup(&new_vec, v->v);

//    GrB_Vector_free(&v->v);
//    GrB_Vector_new(&v->v, GrB_INT32, newsize);
//    GrB_Vector_dup(&v->v, new_vec);

//    GrB_Vector_free(&new_vec);
//    v->size = newsize;
//}

unsigned int vector_int_get_size(vector *v)
{
    return v->size;
}

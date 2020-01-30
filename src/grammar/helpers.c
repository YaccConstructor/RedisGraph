#include <string.h>
#include <assert.h>

#include "helpers.h"

void str_strip(char *buf) {
    char *p = strchr(buf, '\n');
    if (p != NULL) *p = '\0';
}

void check_info(GrB_Info info) {
    assert(info == GrB_SUCCESS);
}
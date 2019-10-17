#include <string.h>

void str_strip(char *buf) {
    char *p = strchr(buf, '\n');
    if (p != NULL) *p = '\0';
}

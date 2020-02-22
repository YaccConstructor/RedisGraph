#include <stdio.h>
#include <assert.h>

#include "index.h"
#include "../util/arr.h"

PathIndex PathIndex_Identity = {
        .left = 0,
        .right = 0,
        .middle = 0,
        .height = 0,
        .length = 0,
};

// Identity = невыводимый путь

void PathIndex_Init(PathIndex *index, uint32_t left, uint32_t right, uint32_t middle, uint32_t height, uint32_t length) {
    index->left = left;
    index->right = right;
    index->middle = middle;
    index->height = height;
    index->length = length;
}

void PathIndex_InitIdentity(PathIndex *index) {
    index->left = 0;
    index->right = 0;
    index->middle = 0;
    index->height = 0;
    index->length = 0;
}

bool PathIndex_IsIdentity(const PathIndex *index) {
    return index->left == 0  &&
           index->right == 0 &&
           index->middle == 0 &&
           index->height == 0 &&
           index->length == 0;
}

bool PathIndex_IsEdge(const PathIndex *index) {
    return index->length == 1;
}

void PathIndex_Copy(const PathIndex *from, PathIndex *to) {
    to->left = from->left;
    to->right = from->right;
    to->middle = from->middle;
    to->height = from->height;
    to->length = from->length;
}

void PathIndex_Mul(void *z, const void *x, const void *y) {
    PathIndex *left = (PathIndex *) x;
    PathIndex *right = (PathIndex *) y;
    PathIndex *res = (PathIndex *) z;

    if (!PathIndex_IsIdentity(left) && !PathIndex_IsIdentity(right)) {
        uint32_t height = (left->height < right->height ? right->height : left->height) + 1;
        PathIndex_Init(res, left->left, right->right, left->right,
                       height, left->length + right->length);

    } else {
        PathIndex_InitIdentity(res);
    }
}

void PathIndex_Add(void *z, const void *x, const void *y) {
    const PathIndex *left = (const PathIndex *) x;
    const PathIndex *right = (const PathIndex *) y;
    PathIndex *res = (PathIndex *) z;

    if (!PathIndex_IsIdentity(left) && !PathIndex_IsIdentity(right)) {
        const PathIndex *min_height_index = (left->height < right->height) ? left : right;
        PathIndex_Copy(min_height_index, res);
    } else if (PathIndex_IsIdentity(left)) {
        PathIndex_Copy(right, res);
    } else {
        PathIndex_Copy(left, res);
    }
}


void PathIndex_ToStr(const PathIndex *index, char *buf) {
    if (PathIndex_IsIdentity(index)) {
        sprintf(buf, "(     Identity      )");
    } else {
        sprintf(buf, "(i:%d,j:%d,k:%d,h=%d,l=%d)",
                index->left, index->right, index->middle, index->height, index->length);
    }
}


void PathIndex_MatrixInit(GrB_Matrix *m, const GrB_Matrix *bool_m) {
    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, *bool_m);

    GrB_Index *I = malloc(nvals * sizeof(GrB_Index));
    GrB_Index *J = malloc(nvals * sizeof(GrB_Index));
    bool *X = malloc(nvals * sizeof(bool));

    GrB_Matrix_extractTuples_BOOL(I, J, X, &nvals, *bool_m);
    for (int k = 0; k < nvals; ++k) {
        PathIndex index;
        PathIndex_Init(&index, I[k], J[k], 0, 1, 1);
        GrB_Matrix_setElement(*m, (void *) &index, I[k], J[k]);
    }

    free(I);
    free(J);
    free(X);
}

void PathIndex_Show(PathIndex *index) {
    char buf[100];
    PathIndex_ToStr(index, buf);
    printf("%s", buf);
}


void PathIndex_MatrixShow(const GrB_Matrix *matrix) {
    GrB_Index n, m;
    GrB_Matrix_nrows(&n, *matrix);
    GrB_Matrix_ncols(&m, *matrix);


    for (GrB_Index i = 0; i < n; i++) {
        for (GrB_Index j = 0; j < m; j++) {
            char buf[50];
            PathIndex index;
            PathIndex_InitIdentity(&index);

            GrB_Matrix_extractElement((void *) &index, *matrix, i, j);
            PathIndex_ToStr(&index, buf);
            printf("i: %lu, j: %lu, index: %s\n", i, j, buf);
        }
        printf("\n");
    }
}


GrB_Index *PathIndex_MatrixGetPath(const GrB_Matrix *matrices, const Grammar *grammar, GrB_Index left, GrB_Index right, MapperIndex nonterm) {
    PathIndex index;
    PathIndex_InitIdentity(&index);
    GrB_Matrix_extractElement((void *) &index, matrices[nonterm], left, right);

    GrB_Index *path = array_new(GrB_Index, index.length + 1);

    PathIndex *sp = array_new(PathIndex, index.length + 1);
    MapperIndex *nonterms = array_new(MapperIndex, index.length + 1);
    uint8_t *directions = array_new(uint8_t, index.length + 1);

    sp = array_append(sp, index);
    nonterms = array_append(nonterms, nonterm);
    directions = array_append(directions, 0);

    while (array_len(sp) != 0) {
        PathIndex top = sp[array_len(sp)-1];
        nonterm = nonterms[array_len(nonterms)-1];
        uint8_t direction = directions[array_len(directions)-1];

        if (PathIndex_IsEdge(&top) || direction == 2) {
            if (PathIndex_IsEdge(&top))
                path = array_append(path, top.left);

            array_pop(sp);
            array_pop(nonterms);
            array_pop(directions);

            if (array_len(directions) != 0)
                directions[array_len(directions)-1] += 1;
            else
                break;
        } else {
            for (GrB_Index i = 0; i < grammar->complex_rules_count; ++i) {
                MapperIndex nonterm_l = grammar->complex_rules[i].l;
                MapperIndex nonterm_r1 = grammar->complex_rules[i].r1;
                MapperIndex nonterm_r2 = grammar->complex_rules[i].r2;

                if (nonterm == nonterm_l) {
                    PathIndex index_r1, index_r2;
                    PathIndex_InitIdentity(&index_r1);
                    PathIndex_InitIdentity(&index_r2);
                    GrB_Matrix_extractElement((void *) &index_r1, matrices[nonterm_r1], top.left, top.middle);
                    GrB_Matrix_extractElement((void *) &index_r2, matrices[nonterm_r2], top.middle, top.right);

                    if (!PathIndex_IsIdentity(&index_r1) && !PathIndex_IsIdentity(&index_r2)) {
                        uint32_t max_height = index_r1.height < index_r2.height ? index_r2.height : index_r1.height;
                        if (top.height == max_height + 1) {
                            if (direction == 0) {
                                sp = array_append(sp, index_r1);
                                nonterms = array_append(nonterms, nonterm_r1);
                            } else {
                                sp = array_append(sp, index_r2);
                                nonterms = array_append(nonterms, nonterm_r2);
                            }
                            directions = array_append(directions, 0);
                            break;
                        }
                    }
                }
            }
        }
    }
    return path;
}

#pragma once

typedef struct {
    int rss;
    int vms;
    int share;
    int text;
    int lib;
    int data;
    int dt;

} MemInfo;

/* vms and rss - KB */
void mem_usage_tick(MemInfo *memInfo);
void mem_usage_tok(MemInfo *result_delta, MemInfo mem_info_start);
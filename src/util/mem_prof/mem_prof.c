#include "mem_prof.h"
#include <stdio.h>
#include <unistd.h>

void mem_usage_tick(MemInfo *memInfo) {
    FILE *f = fopen("/proc/self/statm", "r");
    printf("pid: %d\n", getpid());

    fscanf(f,"%d %d %d %d %d %d %d",
           &memInfo->vms,
           &memInfo->rss,
           &memInfo->share,
           &memInfo->text,
           &memInfo->lib,
           &memInfo->data,
           &memInfo->dt);
		   
	fclose(f);
}

void mem_usage_tok(MemInfo *result_delta, MemInfo mem_info_start) {
    MemInfo mem_info_end;
    mem_usage_tick(&mem_info_end);

    result_delta->vms = mem_info_end.vms - mem_info_start.vms;
    result_delta->rss = mem_info_end.rss - mem_info_start.rss;
    result_delta->share = mem_info_end.share - mem_info_start.share;
    result_delta->text = mem_info_end.text - mem_info_start.text;
    result_delta->lib = mem_info_end.lib - mem_info_start.lib;
    result_delta->data = mem_info_end.data - mem_info_start.data;
    result_delta->dt = mem_info_end.dt - mem_info_start.dt;
}
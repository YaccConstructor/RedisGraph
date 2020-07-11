#include <assert.h>
#include "response.h"

void CfpqResponse_Init(CfpqResponse *resp) {
    resp->iteration_count = 0;
    resp->rss_dif = 0;
    resp->vms_dif = 0;
    resp->count = 0;
    resp->customResp = NULL;
}

int CfpqResponse_Append(CfpqResponse *resp, const char* nonterm, GrB_Index control_sum) {
    assert(resp->count != MAX_NONTERM_COUNT);
    strcpy(resp->nonterms[resp->count], nonterm);
    resp->control_sums[resp->count] = control_sum;
    return resp->count++;
}


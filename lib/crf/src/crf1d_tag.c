/*
 *      CRF1d tagger.
 *
 * Copyright (c) 2007-2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

#include "crf1d.h"

struct tag_crf1mt {
    int num_labels;         /**< Number of distinct output labels (L). */
    int num_attributes;     /**< Number of distinct attributes (A). */

    crf1dm_t *model;        /**< CRF model. */
    crf1d_context_t *ctx;   /**< CRF context. */
};


static void state_score(crf1dt_t* tagger, const crf_instance_t* seq)
{
    int a, i, l, t, r, fid;
    crf1dm_feature_t f;
    feature_refs_t attr;
    floatval_t scale, *state = NULL;
    crf1dm_t* model = tagger->model;
    crf1d_context_t* ctx = tagger->ctx;
    const crf_item_t* item = NULL;
    const int T = seq->num_items;
    const int L = tagger->num_labels;

    /* Loop over the items in the sequence. */
    for (t = 0;t < T;++t) {
        item = &seq->items[t];
        state = STATE_SCORE(ctx, t);

        /* Loop over the contents (attributes) attached to the item. */
        for (i = 0;i < item->num_contents;++i) {
            /* Access the list of state features associated with the attribute. */
            a = item->contents[i].aid;
            crf1dm_get_attrref(model, a, &attr);
            /* A scale usually represents the atrribute frequency in the item. */
            scale = item->contents[i].scale;

            /* Loop over the state features associated with the attribute. */
            for (r = 0;r < attr.num_features;++r) {
                /* The state feature #(attr->fids[r]), which is represented by
                   the attribute #a, outputs the label #(f->dst). */
                fid = crf1dm_get_featureid(&attr, r);
                crf1dm_get_feature(model, fid, &f);
                l = f.dst;
                state[l] += f.weight * scale;
            }
        }
    }
}

static void transition_score(crf1dt_t* tagger)
{
    int i, r, fid;
    crf1dm_feature_t f;
    feature_refs_t edge;
    floatval_t *trans = NULL;
    crf1dm_t* model = tagger->model;
    crf1d_context_t* ctx = tagger->ctx;
    const int L = tagger->num_labels;

    /* Compute transition scores between two labels. */
    for (i = 0;i < L;++i) {
        trans = TRANS_SCORE(ctx, i);
        crf1dm_get_labelref(model, i, &edge);
        for (r = 0;r < edge.num_features;++r) {
            /* Transition feature from #i to #(f->dst). */
            fid = crf1dm_get_featureid(&edge, r);
            crf1dm_get_feature(model, fid, &f);
            trans[f.dst] = f.weight;
        }        
    }
}

crf1dt_t *crf1dt_new(crf1dm_t* crf1dm)
{
    crf1dt_t* crf1dt = NULL;

    crf1dt = (crf1dt_t*)calloc(1, sizeof(crf1dt_t));
    if (crf1dt != NULL) {
        crf1dt->num_labels = crf1dm_get_num_labels(crf1dm);
        crf1dt->num_attributes = crf1dm_get_num_attrs(crf1dm);
        crf1dt->model = crf1dm;
        crf1dt->ctx = crf1dc_new(CTXF_VITERBI, crf1dt->num_labels, 0);
        if (crf1dt->ctx != NULL) {
            crf1dc_reset(crf1dt->ctx, RF_TRANS);
            transition_score(crf1dt);
        } else {
            crf1dt_delete(crf1dt);
            crf1dt = NULL;
        }
    }

    return crf1dt;
}

void crf1dt_delete(crf1dt_t* crf1dt)
{
    if (crf1dt->ctx != NULL) {
        crf1dc_delete(crf1dt->ctx);
        crf1dt->ctx = NULL;
    }
    free(crf1dt);
}

int crf1dt_tag(crf1dt_t* crf1dt, crf_instance_t *inst, int *labels, floatval_t *ptr_score)
{
    int i;
    floatval_t score = 0;
    crf1d_context_t* ctx = crf1dt->ctx;

    crf1dc_set_num_items(ctx, inst->num_items);

    crf1dc_reset(crf1dt->ctx, RF_STATE);
    state_score(crf1dt, inst);
    score = crf1dc_viterbi(ctx);

    for (i = 0;i < inst->num_items;++i) {
        labels[i] = ctx->labels[i];
    }
    if (ptr_score != NULL) {
        *ptr_score = score;
    }

    return 0;
}

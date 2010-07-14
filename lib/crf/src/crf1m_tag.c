/*
 *      Linear-chain CRF tagger.
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

#include "crf1m.h"

struct tag_crf1mt {
    int num_labels;            /**< Number of distinct output labels (L). */
    int num_attributes;        /**< Number of distinct attributes (A). */

    crf1mm_t *model;        /**< CRF model. */
    crf1m_context_t *ctx;    /**< CRF context. */
};


static void state_score(crf1mt_t* tagger, const crf_sequence_t* seq)
{
    int a, i, l, t, r, fid;
    crf1mm_feature_t f;
    feature_refs_t attr;
    floatval_t scale, *state = NULL;
    crf1mm_t* model = tagger->model;
    crf1m_context_t* ctx = tagger->ctx;
    const crf_item_t* item = NULL;
    const int T = seq->num_items;
    const int L = tagger->num_labels;

    /* Loop over the items in the sequence. */
    for (t = 0;t < T;++t) {
        item = &seq->items[t];
        state = STATE_SCORE_AT(ctx, t);

        /* Initialize the state scores at position #t as zero. */
        for (i = 0;i < L;++i) {
            state[i] = 0;
        }

        /* Loop over the contents (attributes) attached to the item. */
        for (i = 0;i < item->num_contents;++i) {
            /* Access the list of state features associated with the attribute. */
            a = item->contents[i].aid;
            crf1mm_get_attrref(model, a, &attr);
            /* A scale usually represents the atrribute frequency in the item. */
            scale = item->contents[i].scale;

            /* Loop over the state features associated with the attribute. */
            for (r = 0;r < attr.num_features;++r) {
                /* The state feature #(attr->fids[r]), which is represented by
                   the attribute #a, outputs the label #(f->dst). */
                fid = crf1mm_get_featureid(&attr, r);
                crf1mm_get_feature(model, fid, &f);
                l = f.dst;
                state[l] += f.weight * scale;
            }
        }
    }
}

static void transition_score(crf1mt_t* tagger)
{
    int i, j, r, fid;
    crf1mm_feature_t f;
    feature_refs_t edge;
    floatval_t *trans = NULL;
    crf1mm_t* model = tagger->model;
    crf1m_context_t* ctx = tagger->ctx;
    const int L = tagger->num_labels;

    /* Initialize all transition scores as zero. */
    for (i = 0;i <= L;++i) {
        trans = TRANS_SCORE_FROM(ctx, i);
        for (j = 0;j <= L;++j) {
            trans[j] = 0;
        }
    }

    /* Compute transition scores from BOS to labels. */
    trans = TRANS_SCORE_FROM(ctx, L);
    crf1mm_get_labelref(model, L, &edge);
    for (r = 0;r < edge.num_features;++r) {
        /* Transition feature from BOS to #(f->dst). */
        fid = crf1mm_get_featureid(&edge, r);
        crf1mm_get_feature(model, fid, &f);
        trans[f.dst] = f.weight;
    }

    /* Compute transition scores between two labels. */
    for (i = 0;i < L;++i) {
        trans = TRANS_SCORE_FROM(ctx, i);
        crf1mm_get_labelref(model, i, &edge);
        for (r = 0;r < edge.num_features;++r) {
            /* Transition feature from #i to #(f->dst). */
            fid = crf1mm_get_featureid(&edge, r);
            crf1mm_get_feature(model, fid, &f);
            trans[f.dst] = f.weight;
        }        
    }

    /* Compute transition scores from labels to EOS. */
    crf1mm_get_labelref(model, L+1, &edge);
    for (r = 0;r < edge.num_features;++r) {
        /* Transition feature from #(f->src) to EOS. */
        fid = crf1mm_get_featureid(&edge, r);
        crf1mm_get_feature(model, fid, &f);
        trans[L] = f.weight;
    }
}

crf1mt_t *crf1mt_new(crf1mm_t* crf1mm)
{
    crf1mt_t* crf1mt = NULL;

    crf1mt = (crf1mt_t*)calloc(1, sizeof(crf1mt_t));
    crf1mt->num_labels = crf1mm_get_num_labels(crf1mm);
    crf1mt->num_attributes = crf1mm_get_num_attrs(crf1mm);
    crf1mt->model = crf1mm;
    crf1mt->ctx = crf1mc_new(crf1mt->num_labels, 0);
    transition_score(crf1mt);

    return crf1mt;
}

void crf1mt_delete(crf1mt_t* crf1mt)
{
    crf1mc_delete(crf1mt->ctx);
    free(crf1mt);
}

int crf1mt_tag(crf1mt_t* crf1mt, crf_sequence_t *inst, crf_output_t* output)
{
    int i;
    floatval_t score = 0;
    crf1m_context_t* ctx = crf1mt->ctx;

    crf1mc_set_num_items(ctx, inst->num_items);

    state_score(crf1mt, inst);
    score = crf1mc_viterbi(ctx);

    crf_output_init_n(output, inst->num_items);
    output->probability = score;
    for (i = 0;i < inst->num_items;++i) {
        output->labels[i] = ctx->labels[i];
    }
    output->num_labels = inst->num_items;

    return 0;
}

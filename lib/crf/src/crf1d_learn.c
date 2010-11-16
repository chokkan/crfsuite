/*
 *      Training routines for CRF1d.
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

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>

#include <crfsuite.h>
#include "crfsuite_internal.h"
#include "crf1d.h"
#include "params.h"
#include "logging.h"

/**
 * Parameters for feature generation.
 */
typedef struct {
    floatval_t  feature_minfreq;                /** The threshold for occurrences of features. */
    int         feature_possible_states;        /** Dense state features. */
    int         feature_possible_transitions;   /** Dense transition features. */
} crf1dl_option_t;

/**
 * CRF1d internal data.
 */
typedef struct {
    int num_labels;                 /**< Number of distinct output labels (L). */
    int num_attributes;             /**< Number of distinct attributes (A). */

    int num_sequences;              /**< The number of instances (N). */
    const crf_instance_t* seqs;     /**< Data set (an array of instances) [N]. */
    int cap_items;                  /**< Maximum length of sequences in the data set. */

    int num_features;               /**< Number of distinct features (K). */
    crf1df_feature_t *features;     /**< Array of feature descriptors [K]. */
    feature_refs_t* attributes;     /**< References to attribute features [A]. */
    feature_refs_t* forward_trans;  /**< References to transition features [L]. */

    crf1d_context_t *ctx;           /**< CRF1d context. */
    crf1dl_option_t opt;            /**< CRF1d options. */

} crf1dl_t;

typedef struct {
    crf1dl_t crf1dt;
} data_internal_t;

typedef struct {
    crf1dl_t crf1dt;
} online_internal_t;


#define    FEATURE(trainer, k) \
    (&(trainer)->features[(k)])
#define    ATTRIBUTE(trainer, a) \
    (&(trainer)->attributes[(a)])
#define    TRANSITION(trainer, i) \
    (&(trainer)->forward_trans[(i)])



/**
 * Initializes the trainer.
 *  @param  trainer     The trainer.
 */
static void
crf1dl_init(
    crf1dl_t *trainer
    )
{
    memset(trainer, 0, sizeof(*trainer));
}

/**
 * Uninitializes the trainer.
 *  @param  trainer     The trainer.
 */
static void
crf1dl_finish(
    crf1dl_t *trainer
    )
{
    if (trainer->ctx != NULL) {
        crf1dc_delete(trainer->ctx);
        trainer->ctx = NULL;
    }
    if (trainer->features != NULL) {
        free(trainer->features);
        trainer->features = NULL;
    }
    if (trainer->attributes != NULL) {
        free(trainer->attributes);
        trainer->attributes = NULL;
    }
    if (trainer->forward_trans != NULL) {
        free(trainer->forward_trans);
        trainer->forward_trans = NULL;
    }
}

/**
 * Fills the state score table for the instance.
 *  @param  trainer     The trainer.
 *  @param  seq         The instance.
 *  @param  w           The array of feature weights.
 */
static void
crf1dl_state_score(
    crf1dl_t* trainer,
    const crf_instance_t* seq,
    const floatval_t* w
    )
{
    int i, t, r;
    crf1d_context_t* ctx = trainer->ctx;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

    /* Loop over the items in the sequence. */
    for (t = 0;t < T;++t) {
        const crf_item_t *item = &seq->items[t];
        floatval_t *state = STATE_SCORE(ctx, t);

        /* Loop over the contents (attributes) attached to the item. */
        for (i = 0;i < item->num_contents;++i) {
            /* Access the list of state features associated with the attribute. */
            int a = item->contents[i].aid;
            const feature_refs_t *attr = ATTRIBUTE(trainer, a);
            floatval_t value = item->contents[i].scale;

            /* Loop over the state features associated with the attribute. */
            for (r = 0;r < attr->num_features;++r) {
                /* State feature associates the attribute #a with the label #(f->dst). */
                int fid = attr->fids[r];
                const crf1df_feature_t *f = FEATURE(trainer, fid);
                state[f->dst] += w[fid] * value;
            }
        }
    }
}

/**
 * Fills the state score table for the instance (with scaling).
 *  @param  trainer     The trainer.
 *  @param  seq         The instance.
 *  @param  w           The array of feature weights.
 *  @param  scale       The scale factor for the feature weights.
 */
static void
crf1dl_state_score_scaled(
    crf1dl_t* trainer,
    const crf_instance_t* seq,
    const floatval_t* w,
    floatval_t scale
    )
{
    int i, t, r;
    crf1d_context_t* ctx = trainer->ctx;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

    /* Loop over the items in the sequence. */
    for (t = 0;t < T;++t) {
        const crf_item_t *item = &seq->items[t];
        floatval_t *state = STATE_SCORE(ctx, t);

        /* Loop over the contents (attributes) attached to the item. */
        for (i = 0;i < item->num_contents;++i) {
            /* Access the list of state features associated with the attribute. */
            int a = item->contents[i].aid;
            const feature_refs_t *attr = ATTRIBUTE(trainer, a);
            floatval_t value = item->contents[i].scale * scale;

            /* Loop over the state features associated with the attribute. */
            for (r = 0;r < attr->num_features;++r) {
                /* State feature associates the attribute #a with the label #(f->dst). */
                int fid = attr->fids[r];
                const crf1df_feature_t *f = FEATURE(trainer, fid);
                state[f->dst] += w[fid] * value;
            }
        }
    }
}

/**
 * Fills the transition score table for the instance.
 *  @param  trainer     The trainer.
 *  @param  w           The array of feature weights.
 */
static void
crf1dl_transition_score(
    crf1dl_t* trainer,
    const floatval_t* w
    )
{
    int i, r;
    crf1d_context_t* ctx = trainer->ctx;
    const int L = trainer->num_labels;

    /* Compute transition scores between two labels. */
    for (i = 0;i < L;++i) {
        floatval_t *trans = TRANS_SCORE(ctx, i);
        const feature_refs_t *edge = TRANSITION(trainer, i);
        for (r = 0;r < edge->num_features;++r) {
            /* Transition feature from #i to #(f->dst). */
            int fid = edge->fids[r];
            const crf1df_feature_t *f = FEATURE(trainer, fid);
            trans[f->dst] = w[fid];
        }        
    }
}

/**
 * Fills the transition score table for the instance (with scaling).
 *  @param  trainer     The trainer.
 *  @param  w           The array of feature weights.
 *  @param  scale       The scale factor for the feature weights.
 */
static void
crf1dl_transition_score_scaled(
    crf1dl_t* trainer,
    const floatval_t* w,
    floatval_t scale
    )
{
    int i, r;
    crf1d_context_t* ctx = trainer->ctx;
    const int L = trainer->num_labels;

    /* Compute transition scores between two labels. */
    for (i = 0;i < L;++i) {
        floatval_t *trans = TRANS_SCORE(ctx, i);
        const feature_refs_t *edge = TRANSITION(trainer, i);
        for (r = 0;r < edge->num_features;++r) {
            /* Transition feature from #i to #(f->dst). */
            int fid = edge->fids[r];
            const crf1df_feature_t *f = FEATURE(trainer, fid);
            trans[f->dst] = w[fid] * scale;
        }        
    }
}

static void
crf1dl_enum_features(
    crf1dl_t *trainer,
    const crf_instance_t *seq,
    const int *labels,
    crf_train_enum_features_callback func,
    void *instance
    )
{
    int c, i = -1, t, r;
    crf1d_context_t* ctx = trainer->ctx;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

    /* Loop over the items in the sequence. */
    for (t = 0;t < T;++t) {
        const crf_item_t *item = &seq->items[t];
        const int j = labels[t];

        /* Loop over the contents (attributes) attached to the item. */
        for (c = 0;c < item->num_contents;++c) {
            /* Access the list of state features associated with the attribute. */
            int a = item->contents[c].aid;
            const feature_refs_t *attr = ATTRIBUTE(trainer, a);
            floatval_t scale = item->contents[c].scale;

            /* Loop over the state features associated with the attribute. */
            for (r = 0;r < attr->num_features;++r) {
                /* State feature associates the attribute #a with the label #(f->dst). */
                int fid = attr->fids[r];
                const crf1df_feature_t *f = FEATURE(trainer, fid);
                if (f->dst == j) {
                    func(instance, fid, scale);
                }
            }
        }

        if (i != -1) {
            const feature_refs_t *edge = TRANSITION(trainer, i);
            for (r = 0;r < edge->num_features;++r) {
                /* Transition feature from #i to #(f->dst). */
                int fid = edge->fids[r];
                const crf1df_feature_t *f = FEATURE(trainer, fid);
                if (f->dst == j) {
                    func(instance, fid, 1.);
                }
            }
        }

        i = j;
    }
}

/**
 * Accumulates the model expectations of features.
 *  @param  trainer     The trainer.
 *  @param  seq         The instance.
 *  @param  w           The array to which the model expectations are accumulated.
 */
static void
crf1dl_model_expectation(
    crf1dl_t* trainer,
    const crf_instance_t* seq,
    floatval_t *w
    )
{
    int a, c, i, t, r;
    crf1d_context_t* ctx = trainer->ctx;
    const feature_refs_t *attr = NULL, *trans = NULL;
    const crf_item_t* item = NULL;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

    for (t = 0;t < T;++t) {
        floatval_t *prob = STATE_MEXP(ctx, t);

        /* Compute expectations for state features at position #t. */
        item = &seq->items[t];
        for (c = 0;c < item->num_contents;++c) {
            /* Access the attribute. */
            floatval_t scale = item->contents[c].scale;
            a = item->contents[c].aid;
            attr = ATTRIBUTE(trainer, a);

            /* Loop over state features for the attribute. */
            for (r = 0;r < attr->num_features;++r) {
                int fid = attr->fids[r];
                crf1df_feature_t *f = FEATURE(trainer, fid);
                w[fid] += prob[f->dst] * scale;
            }
        }
    }

    /* Loop over the labels (t, i) */
    for (i = 0;i < L;++i) {
        const floatval_t *prob = TRANS_MEXP(ctx, i);
        const feature_refs_t *edge = TRANSITION(trainer, i);
        for (r = 0;r < edge->num_features;++r) {
            /* Transition feature from #i to #(f->dst). */
            int fid = edge->fids[r];
            crf1df_feature_t *f = FEATURE(trainer, fid);
            w[fid] += prob[f->dst];
        }
    }
}

/**
 * Tags an instance.
 *  @param  trainer     The trainer.
 *  @param  w           The array of feature weights.
 *  @param  seq         The instance to be tagged.
 *  @param  output      The object that receives the predicted label sequence.
 */
static int
crf1dl_tag(
    crf1dl_t* trainer,
    const floatval_t *w,
    const crf_instance_t *seq,
    int *viterbi,
    floatval_t *ptr_score
    )
{
    int i;
    floatval_t score = 0;
    const int T = seq->num_items;
    crf1d_context_t* ctx = trainer->ctx;
    
    crf1dc_reset(trainer->ctx, RF_TRANS | RF_STATE);
    crf1dl_transition_score(trainer, w);
    crf1dc_set_num_items(trainer->ctx, T);
    crf1dl_state_score(trainer, seq, w);
    score = crf1dc_viterbi(trainer->ctx);

    for (i = 0;i < T;++i) {
        viterbi[i] = trainer->ctx->labels[i];
    }
    if (ptr_score != NULL) {
        *ptr_score = score;
    }
    return 0;
}

/**
 * Initializes, loads, or stores parameter values from/to a parameter object.
 *  @param  params      The parameter object.
 *  @param  opt         Pointer to the option structure.
 *  @param  mode        Update mode.
 */
static int crf1dl_exchange_options(crf_params_t* params, crf1dl_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_FLOAT(
            "feature.minfreq", opt->feature_minfreq, 0.0,
            "The minimum frequency of features."
            )
        DDX_PARAM_INT(
            "feature.possible_states", opt->feature_possible_states, 0,
            "Force to generate possible state features."
            )
        DDX_PARAM_INT(
            "feature.possible_transitions", opt->feature_possible_transitions, 0,
            "Force to generate possible transition features."
            )
    END_PARAM_MAP()

    return 0;
}



static int
crf1dl_set_data(
    crf1dl_t *crf1dt,
    const crf_instance_t *seqs,
    int num_instances,
    int num_labels,
    int num_attributes,
    logging_t *lg
    )
{
    int i, ret = 0;
    clock_t begin = 0;
    int T = 0;
    const int L = num_labels;
    const int A = num_attributes;
    const int N = num_instances;
    crf1dl_option_t *opt = &crf1dt->opt;

    crf1dt->seqs = seqs;
    crf1dt->num_sequences = N;
    crf1dt->num_attributes = A;
    crf1dt->num_labels = L;

    /* Find the maximum length of items. */
    for (i = 0;i < N;++i) {
        if (T < seqs[i].num_items) {
            T = seqs[i].num_items;
        }
    }

    /* Construct a CRF context. */
    crf1dt->ctx = crf1dc_new(CTXF_MARGINALS | CTXF_VITERBI, L, T);
    if (crf1dt->ctx == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Feature generation. */
    logging(lg, "Feature generation\n");
    logging(lg, "type: CRF1d\n");
    logging(lg, "feature.minfreq: %f\n", opt->feature_minfreq);
    logging(lg, "feature.possible_states: %d\n", opt->feature_possible_states);
    logging(lg, "feature.possible_transitions: %d\n", opt->feature_possible_transitions);
    begin = clock();
    crf1dt->features = crf1df_generate(
        &crf1dt->num_features,
        seqs,
        N,
        L,
        A,
        opt->feature_possible_states ? 1 : 0,
        opt->feature_possible_transitions ? 1 : 0,
        opt->feature_minfreq,
        lg->func,
        lg->instance
        );
    if (crf1dt->features == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    logging(lg, "Number of features: %d\n", crf1dt->num_features);
    logging(lg, "Seconds required: %.3f\n", (clock() - begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    /* Initialize the feature references. */
    crf1df_init_references(
        &crf1dt->attributes,
        &crf1dt->forward_trans,
        crf1dt->features,
        crf1dt->num_features,
        A,
        L);
    if (crf1dt->attributes == NULL || crf1dt->forward_trans == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    return ret;

error_exit:
    crf1dl_finish(crf1dt);
    return ret;
}

static int
crf1dl_save_model(
    crf1dl_t *crf1dt,
    const char *filename,
    const floatval_t *w,
    crf_dictionary_t *attrs,
    crf_dictionary_t *labels,
    logging_t *lg
    )
{
    int a, k, l, ret;
    clock_t begin;
    int *fmap = NULL, *amap = NULL;
    crf1dmw_t* writer = NULL;
    const feature_refs_t *edge = NULL, *attr = NULL;
    const floatval_t threshold = 0.01;
    const int L = crf1dt->num_labels;
    const int A = crf1dt->num_attributes;
    const int K = crf1dt->num_features;
    int J = 0, B = 0;

    /* Start storing the model. */
    logging(lg, "Storing the model\n");
    begin = clock();

    /* Allocate and initialize the feature mapping. */
    fmap = (int*)calloc(K, sizeof(int));
    if (fmap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (k = 0;k < K;++k) fmap[k] = k;
    J = K;
#else
    for (k = 0;k < K;++k) fmap[k] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /* Allocate and initialize the attribute mapping. */
    amap = (int*)calloc(A, sizeof(int));
    if (amap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (a = 0;a < A;++a) amap[a] = a;
    B = A;
#else
    for (a = 0;a < A;++a) amap[a] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /*
     *    Open a model writer.
     */
    writer = crf1mmw(filename);
    if (writer == NULL) {
        goto error_exit;
    }

    /* Open a feature chunk in the model file. */
    if (ret = crf1dmw_open_features(writer)) {
        goto error_exit;
    }

    /* Determine a set of active features and attributes. */
    for (k = 0;k < K;++k) {
        crf1df_feature_t* f = &crf1dt->features[k];
        if (w[k] != 0) {
            int src;
            crf1dm_feature_t feat;

#ifndef    CRF_TRAIN_SAVE_NO_PRUNING
            /* The feature (#k) will have a new feature id (#J). */
            fmap[k] = J++;        /* Feature #k -> #fmap[k]. */

            /* Map the source of the field. */
            if (f->type == FT_STATE) {
                /* The attribute #(f->src) will have a new attribute id (#B). */
                if (amap[f->src] < 0) amap[f->src] = B++;    /* Attribute #a -> #amap[a]. */
                src = amap[f->src];
            } else {
                src = f->src;
            }
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

            feat.type = f->type;
            feat.src = src;
            feat.dst = f->dst;
            feat.weight = w[k];

            /* Write the feature. */
            if (ret = crf1dmw_put_feature(writer, fmap[k], &feat)) {
                goto error_exit;
            }
        }
    }

    /* Close the feature chunk. */
    if (ret = crf1dmw_close_features(writer)) {
        goto error_exit;
    }

    logging(lg, "Number of active features: %d (%d)\n", J, K);
    logging(lg, "Number of active attributes: %d (%d)\n", B, A);
    logging(lg, "Number of active labels: %d (%d)\n", L, L);

    /* Write labels. */
    logging(lg, "Writing labels\n", L);
    if (ret = crf1dmw_open_labels(writer, L)) {
        goto error_exit;
    }
    for (l = 0;l < L;++l) {
        const char *str = NULL;
        labels->to_string(labels, l, &str);
        if (str != NULL) {
            if (ret = crf1dmw_put_label(writer, l, str)) {
                goto error_exit;
            }
            labels->free(labels, str);
        }
    }
    if (ret = crf1dmw_close_labels(writer)) {
        goto error_exit;
    }

    /* Write attributes. */
    logging(lg, "Writing attributes\n");
    if (ret = crf1dmw_open_attrs(writer, B)) {
        goto error_exit;
    }
    for (a = 0;a < A;++a) {
        if (0 <= amap[a]) {
            const char *str = NULL;
            attrs->to_string(attrs, a, &str);
            if (str != NULL) {
                if (ret = crf1dmw_put_attr(writer, amap[a], str)) {
                    goto error_exit;
                }
                attrs->free(attrs, str);
            }
        }
    }
    if (ret = crf1dmw_close_attrs(writer)) {
        goto error_exit;
    }

    /* Write label feature references. */
    logging(lg, "Writing feature references for transitions\n");
    if (ret = crf1dmw_open_labelrefs(writer, L+2)) {
        goto error_exit;
    }
    for (l = 0;l < L;++l) {
        edge = TRANSITION(crf1dt, l);
        if (ret = crf1dmw_put_labelref(writer, l, edge, fmap)) {
            goto error_exit;
        }
    }
    if (ret = crf1dmw_close_labelrefs(writer)) {
        goto error_exit;
    }

    /* Write attribute feature references. */
    logging(lg, "Writing feature references for attributes\n");
    if (ret = crf1dmw_open_attrrefs(writer, B)) {
        goto error_exit;
    }
    for (a = 0;a < A;++a) {
        if (0 <= amap[a]) {
            attr = ATTRIBUTE(crf1dt, a);
            if (ret = crf1dmw_put_attrref(writer, amap[a], attr, fmap)) {
                goto error_exit;
            }
        }
    }
    if (ret = crf1dmw_close_attrrefs(writer)) {
        goto error_exit;
    }

    /* Close the writer. */
    crf1dmw_close(writer);
    logging(lg, "Seconds required: %.3f\n", (clock() - begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    free(amap);
    free(fmap);
    return 0;

error_exit:
    if (writer != NULL) {
        crf1dmw_close(writer);
    }
    if (amap != NULL) {
        free(amap);
    }
    if (fmap != NULL) {
        free(fmap);
    }
    return ret;
}




static int crf1dl_batch_exchange_options(crf_train_data_t *self, crf_params_t* params, int mode)
{
    data_internal_t *batch = (data_internal_t*)self->internal;
    return crf1dl_exchange_options(params, &batch->crf1dt.opt, mode);
}

static int crf1dl_batch_set_data(
    crf_train_data_t *self,
    const crf_instance_t *seqs,
    int num_instances,
    crf_dictionary_t *attrs,
    crf_dictionary_t *labels,
    logging_t *lg
    )
{
    data_internal_t *batch = (data_internal_t*)self->internal;
    int ret = crf1dl_set_data(&batch->crf1dt, seqs, num_instances, labels->num(labels), attrs->num(attrs), lg);
    self->seqs = seqs;
    self->num_instances = num_instances;
    self->attrs = attrs;
    self->labels = labels;
    self->num_features = batch->crf1dt.num_features;
    self->cap_items = batch->crf1dt.ctx->cap_items;
    return ret;
}

static int crf1dl_batch_holdout_evaluation(crf_train_data_t *self, const floatval_t *w)
{
    int i;
    crf_evaluation_t eval;
    data_internal_t *batch = (data_internal_t*)self->internal;
    crf1dl_t* crf1dt = &batch->crf1dt;
    const crf_instance_t* seqs = crf1dt->seqs;
    const int N = crf1dt->num_sequences;

    /* Initialize the evaluation table. */
    crf_evaluation_init(&eval, self->labels->num(self->labels));

    /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
     */
    crf1dc_reset(crf1dt->ctx, RF_TRANS);
    crf1dl_transition_score(crf1dt, w);

    for (i = 0;i < N;++i) {
        const crf_instance_t* seq = &seqs[i];

        /* Use holdout data only. */
        if (seq->group != self->holdout) {
            continue;
        }

        /* Set label sequences and state scores. */
        crf1dc_set_num_items(crf1dt->ctx, seq->num_items);
        crf1dc_reset(crf1dt->ctx, RF_STATE);
        crf1dl_state_score(crf1dt, seq, w);

        /* Tag the current instance. */
        crf1dc_viterbi(crf1dt->ctx);

        crf_evaluation_accmulate(&eval, seq, crf1dt->ctx->labels);
    }

    /* Report the performance. */
    crf_evaluation_compute(&eval);
    crf_evaluation_output(&eval, self->labels, stdout);
}

static int crf1dl_batch_objective_and_gradients(crf_train_data_t *self, const floatval_t *w, floatval_t *f, floatval_t *g)
{
    int i;
    floatval_t logp = 0, logl = 0;
    data_internal_t *batch = (data_internal_t*)self->internal;
    crf1dl_t* crf1dt = &batch->crf1dt;
    const crf_instance_t* seqs = crf1dt->seqs;
    const int N = crf1dt->num_sequences;
    const int K = crf1dt->num_features;

    /*
        Initialize the gradients with observation expectations.
     */
    for (i = 0;i < K;++i) {
        crf1df_feature_t* f = &crf1dt->features[i];
        g[i] = -f->freq;
    }

    /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
     */
    crf1dc_reset(crf1dt->ctx, RF_TRANS);
    crf1dl_transition_score(crf1dt, w);
    crf1dc_exp_transition(crf1dt->ctx);

    /*
        Compute model expectations.
     */
    for (i = 0;i < N;++i) {
        const crf_instance_t* seq = &seqs[i];

        /* Ignore holdout data. */
        if (seq->group == self->holdout) {
            continue;
        }

        /* Set label sequences and state scores. */
        crf1dc_set_num_items(crf1dt->ctx, seq->num_items);
        crf1dc_reset(crf1dt->ctx, RF_STATE);
        crf1dl_state_score(crf1dt, seq, w);
        crf1dc_exp_state(crf1dt->ctx);

        /* Compute forward/backward scores. */
        crf1dc_alpha_score(crf1dt->ctx);
        crf1dc_beta_score(crf1dt->ctx);
        crf1dc_marginal(crf1dt->ctx);

        /* Compute the probability of the input sequence on the model. */
        logp = crf1dc_score(crf1dt->ctx, seq->labels) - crf1dc_lognorm(crf1dt->ctx);
        /* Update the log-likelihood. */
        logl += logp;

        /* Update the model expectations of features. */
        crf1dl_model_expectation(crf1dt, seq, g);
    }

    *f = -logl;
    return 0;
}

static int crf1dl_batch_enum_features(crf_train_data_t *self, const crf_instance_t *seq, const int *labels, crf_train_enum_features_callback func, void *instance)
{
    data_internal_t *batch = (data_internal_t*)self->internal;
    crf1dl_enum_features(&batch->crf1dt, seq, labels, func, instance);
    return 0;
}

static int crf1dl_batch_save_model(crf_train_data_t *self, const char *filename, const floatval_t *w, logging_t *lg)
{
    data_internal_t *batch = (data_internal_t*)self->internal;
    return crf1dl_save_model(&batch->crf1dt, filename, w, self->attrs, self->labels, lg);
}

static int crf1dl_batch_tag(crf_train_data_t *self, const floatval_t *w, const crf_instance_t *inst, int *viterbi, floatval_t *ptr_score)
{
    data_internal_t *batch = (data_internal_t*)self->internal;
    return crf1dl_tag(&batch->crf1dt, w, inst, viterbi, ptr_score);
}

crf_train_data_t *crf1dl_create_instance_batch()
{
    crf_train_data_t* self = (crf_train_data_t*)calloc(1, sizeof(crf_train_data_t));
    if (self != NULL) {
        data_internal_t *batch = (data_internal_t*)calloc(1, sizeof(data_internal_t));
        if (batch != NULL) {
            crf1dl_init(&batch->crf1dt);

            self->exchange_options = crf1dl_batch_exchange_options;
            self->set_data = crf1dl_batch_set_data;
            self->enum_features = crf1dl_batch_enum_features;
            self->objective_and_gradients = crf1dl_batch_objective_and_gradients;
            self->save_model = crf1dl_batch_save_model;
            self->tag = crf1dl_batch_tag;
            self->holdout_evaluation = crf1dl_batch_holdout_evaluation;
            self->internal = batch;
        }
    }

    return self;
}

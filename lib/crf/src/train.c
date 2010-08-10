/*
 *      Implementation of the training interface (crf_trainer_t).
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crfsuite.h>
#include "crfsuite_internal.h"
#include "params.h"
#include "mt19937ar.h"

#include "logging.h"
#include "crf1d.h"




void crf1dl_shuffle(int *perm, int N, int init)
{
    int i, j, tmp;

    if (init) {
        /* Initialize the permutation if necessary. */
        for (i = 0;i < N;++i) {
            perm[i] = i;
        }
    }

    for (i = 0;i < N;++i) {
        j = mt_genrand_int31() % N;
        tmp = perm[j];
        perm[j] = perm[i];
        perm[i] = tmp;
    }
}






static crf_train_internal_t* crf_train_new(const char *interface)
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)calloc(1, sizeof(crf_train_internal_t));
    trainer->lg = (logging_t*)calloc(1, sizeof(logging_t));
    trainer->params = params_create_instance();

    trainer->batch = crf1dl_create_instance_batch();
    trainer->batch->exchange_options(trainer->batch, trainer->params, 0);
    crf_train_lbfgs_init(trainer->params);

    return trainer;
}

static void crf_train_delete(crf_trainer_t* self)
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)self->internal;
    if (trainer != NULL) {
        if (trainer->params != NULL) {
            trainer->params->release(trainer->params);
        }
        free(trainer->lg);
        free(trainer);
    }
}

static int crf_train_addref(crf_trainer_t* trainer)
{
    return crf_interlocked_increment(&trainer->nref);
}

static int crf_train_release(crf_trainer_t* trainer)
{
    int count = crf_interlocked_decrement(&trainer->nref);
    if (count == 0) {
        crf_train_delete(trainer);
    }
    return count;
}

static void crf_train_set_message_callback(crf_trainer_t* self, void *instance, crf_logging_callback cbm)
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)self->internal;
    trainer->lg->func = cbm;
    trainer->lg->instance = instance;
}

static void crf_train_set_evaluate_callback(crf_trainer_t* self, void *instance, crf_evaluate_callback cbe)
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)self->internal;
    trainer->cbe_instance = instance;
    trainer->cbe_proc = cbe;
}

static crf_params_t* crf_train_params(crf_trainer_t* self)
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)self->internal;
    crf_params_t* params = trainer->params;
    params->addref(params);
    return params;
}

static int crf_train_train(
    crf_trainer_t* self,
    const crf_sequence_t* seqs,
    int num_instances,
    crf_dictionary_t* attrs,
    crf_dictionary_t* labels,
    const char *filename
    )
{
    crf_train_internal_t *trainer = (crf_train_internal_t*)self->internal;
    crf_train_batch_t *batch = trainer->batch;
    
    crf_train_lbfgs(
        batch,
        trainer->params,
        trainer->lg,
        seqs,
        num_instances,
        labels->num(labels),
        attrs->num(attrs)
        );

#if 0
    int i, max_item_length;
    int ret = 0;
    floatval_t sigma = 10, *best_w = NULL;
    crf_sequence_t* seqs = (crf_sequence_t*)instances;
    crf1df_features_t* features = NULL;
    crf1dl_t *crf1dt = (crf1dl_t*)trainer->internal;
    crf_params_t *params = crf1dt->params;
    crf1dl_option_t *opt = &crf1dt->opt;

    /* Obtain the maximum number of items. */
    max_item_length = 0;
    for (i = 0;i < num_instances;++i) {
        if (max_item_length < seqs[i].num_items) {
            max_item_length = seqs[i].num_items;
        }
    }

    /* Access parameters. */
    crf1dl_exchange_options(crf1dt->params, opt, -1);

    /* Report the parameters. */
    logging(crf1dt->lg, "Training first-order linear-chain CRFs (trainer.crf1m)\n");
    logging(crf1dt->lg, "\n");

    /* Generate features. */
    logging(crf1dt->lg, "Feature generation\n");
    logging(crf1dt->lg, "feature.minfreq: %f\n", opt->feature_minfreq);
    logging(crf1dt->lg, "feature.possible_states: %d\n", opt->feature_possible_states);
    logging(crf1dt->lg, "feature.possible_transitions: %d\n", opt->feature_possible_transitions);
    crf1dt->clk_begin = clock();
    features = crf1df_generate(
        seqs,
        num_instances,
        num_labels,
        num_attributes,
        opt->feature_possible_states ? 1 : 0,
        opt->feature_possible_transitions ? 1 : 0,
        opt->feature_minfreq,
        crf1dt->lg->func,
        crf1dt->lg->instance
        );
    logging(crf1dt->lg, "Number of features: %d\n", features->num_features);
    logging(crf1dt->lg, "Seconds required: %.3f\n", (clock() - crf1dt->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crf1dt->lg, "\n");

    /* Preparation for training. */
    crf1dl_prepare(crf1dt, num_labels, num_attributes, max_item_length, features);
    crf1dt->num_attributes = num_attributes;
    crf1dt->num_labels = num_labels;
    crf1dt->num_sequences = num_instances;
    crf1dt->seqs = seqs;

    crf1dt->tagger.internal = crf1dt;
    crf1dt->tagger.tag = crf_train_tag;

    if (strcmp(opt->algorithm, "lbfgs") == 0) {
        ret = crf_train_lbfgs(crf1dt, opt);
    /*} else if (strcmp(opt->algorithm, "sgd") == 0) {
        ret = crf1dl_sgd(crf1dt, opt);*/
    } else {
        return CRFERR_INTERNAL_LOGIC;
    }

    return ret;
#endif
    return 0;
}

int crf1dl_create_instance(const char *interface, void **ptr)
{
    if (strcmp(interface, "trainer.crf1m") == 0) {
        crf_trainer_t* trainer = (crf_trainer_t*)calloc(1, sizeof(crf_trainer_t));

        trainer->nref = 1;
        trainer->addref = crf_train_addref;
        trainer->release = crf_train_release;

        trainer->params = crf_train_params;
    
        trainer->set_message_callback = crf_train_set_message_callback;
        trainer->set_evaluate_callback = crf_train_set_evaluate_callback;
        trainer->train = crf_train_train;
        trainer->internal = crf_train_new(interface);

        *ptr = trainer;
        return 0;
    } else {
        return 1;
    }
}

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

static int crf_tag_notimplemented(crf_tagger_t *tagger)
{
    return CRFERR_NOTIMPLEMENTED;
}

static int
crf_tag_tag(crf_tagger_t* tagger, crf_instance_t *inst, int *labels, floatval_t *ptr_score)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)tagger->internal;
    crf_train_data_t *batch = tr->data;
    return CRFERR_NOTIMPLEMENTED;
    //return batch->tag(batch, w, inst, labels, ptr_score);
}

static crf_train_internal_t* crf_train_new(int ftype, int algorithm)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)calloc(
        1, sizeof(crf_train_internal_t));
    if (tr != NULL) {
        tr->lg = (logging_t*)calloc(1, sizeof(logging_t));
        tr->params = params_create_instance();
        tr->feature_type = ftype;
        tr->algorithm = algorithm;

        tr->data = crf1dl_create_instance_batch();
        tr->data->exchange_options(tr->data, tr->params, 0);

        /* Initialize parameters for the training algorithm. */
        switch (algorithm) {
        case TRAIN_LBFGS:
            crf_train_lbfgs_init(tr->params);
            break;
        case TRAIN_AVERAGED_PERCEPTRON:
            crf_train_averaged_perceptron_init(tr->params);
            break;
        }
    }

    return tr;
}

static void crf_train_delete(crf_trainer_t* self)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    if (tr != NULL) {
        if (tr->params != NULL) {
            tr->params->release(tr->params);
        }
        free(tr->lg);
        free(tr);
    }
}

static int crf_train_addref(crf_trainer_t* tr)
{
    return crf_interlocked_increment(&tr->nref);
}

static int crf_train_release(crf_trainer_t* self)
{
    int count = crf_interlocked_decrement(&self->nref);
    if (count == 0) {
        crf_train_delete(self);
    }
    return count;
}

static void crf_train_set_message_callback(crf_trainer_t* self, void *instance, crf_logging_callback cbm)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    tr->lg->func = cbm;
    tr->lg->instance = instance;
}

static void crf_train_set_evaluate_callback(crf_trainer_t* self, void *instance, crf_evaluate_callback cbe)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    tr->cbe_instance = instance;
    tr->cbe_proc = cbe;
}

static crf_params_t* crf_train_params(crf_trainer_t* self)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    crf_params_t* params = tr->params;
    params->addref(params);
    return params;
}

static int crf_train_batch(
    crf_trainer_t* self,
    const crf_instance_t* seqs,
    int num_instances,
    crf_dictionary_t* attrs,
    crf_dictionary_t* labels,
    const char *filename
    )
{
    char *algorithm = NULL;
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    logging_t *lg = tr->lg;
    crf_train_data_t *data = tr->data;
    floatval_t *w = NULL;
    const int N = num_instances;
    const int L = labels->num(labels);
    const int A = attrs->num(attrs);

    /* Show the training algorithm. */
    logging(lg, "Training\n");
    logging(lg, "algorithm: ");
    switch (tr->algorithm) {
    case TRAIN_LBFGS:
        logging(lg, "L-BFGS");
        break;
    case TRAIN_AVERAGED_PERCEPTRON:
        logging(lg, "Averaged Perceptron");
        break;
    }
    logging(lg, "\n");
    logging(lg, "\n");

    /* Set the training set to the CRF, and generate features. */
    data->set_data(data, seqs, N, L, A, lg);

    switch (tr->algorithm) {
    case TRAIN_LBFGS:
        crf_train_lbfgs(
            data,
            tr->params,
            lg,
            &w,
            tr->cbe_proc,
            tr->cbe_instance
            );
        break;
    case TRAIN_AVERAGED_PERCEPTRON:
        crf_train_averaged_perceptron(
            data,
            tr->params,
            lg,
            &w,
            tr->cbe_proc,
            tr->cbe_instance
            );
        break;
    }

    free(w);

    return 0;
}

int crf1dl_create_instance(const char *interface, void **ptr)
{
    int ftype = FTYPE_NONE;
    int algorithm = TRAIN_NONE;

    /* Check if the interface name begins with "train/". */
    if (strncmp(interface, "train/", 6) != 0) {
        return 1;
    }
    interface += 6;

    /* Obtain the feature type. */
    if (strncmp(interface, "crf1d/", 6) == 0) {
        ftype = FTYPE_CRF1D;
        interface += 6;
    } else {
        return 1;
    }

    /* Obtain the training algorithm. */
    if (strcmp(interface, "lbfgs") == 0) {
        algorithm = TRAIN_LBFGS;
    } else if (strcmp(interface, "averaged-perceptron") == 0) {
        algorithm = TRAIN_AVERAGED_PERCEPTRON;
    } else {
        return 1;
    }

    /* Create an instance. */
    if (ftype != FTYPE_NONE && algorithm != TRAIN_NONE) {
        crf_trainer_t* trainer = (crf_trainer_t*)calloc(1, sizeof(crf_trainer_t));
        if (trainer != NULL) {
            trainer->internal = crf_train_new(ftype, algorithm);
            if (trainer->internal != NULL) {
                trainer->nref = 1;
                trainer->addref = crf_train_addref;
                trainer->release = crf_train_release;
                trainer->params = crf_train_params;
                trainer->set_message_callback = crf_train_set_message_callback;
                trainer->set_evaluate_callback = crf_train_set_evaluate_callback;
                trainer->train = crf_train_batch;

                *ptr = trainer;
                return 0;
            } else {
                free(trainer);
                trainer = NULL;
            }
        }
    }

    return 1;
}

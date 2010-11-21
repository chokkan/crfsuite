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

#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>
#include "crfsuite_internal.h"
#include "params.h"
#include "logging.h"
#include "crf1d.h"

static crf_train_internal_t* crf_train_new(int ftype, int algorithm)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)calloc(1, sizeof(crf_train_internal_t));
    if (tr != NULL) {
        tr->lg = (logging_t*)calloc(1, sizeof(logging_t));
        tr->params = params_create_instance();
        tr->feature_type = ftype;
        tr->algorithm = algorithm;

        tr->gm = crf1d_create_encoder();
        tr->gm->exchange_options(tr->gm, tr->params, 0);

        /* Initialize parameters for the training algorithm. */
        switch (algorithm) {
        case TRAIN_LBFGS:
            crf_train_lbfgs_init(tr->params);
            break;
        case TRAIN_L2SGD:
            crf_train_l2sgd_init(tr->params);
            break;
        case TRAIN_AVERAGED_PERCEPTRON:
            crf_train_averaged_perceptron_init(tr->params);
            break;
        case TRAIN_PASSIVE_AGGRESSIVE:
            crf_train_passive_aggressive_init(tr->params);
            break;
        case TRAIN_AROW:
            crf_train_arow_init(tr->params);
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

static crf_params_t* crf_train_params(crf_trainer_t* self)
{
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    crf_params_t* params = tr->params;
    params->addref(params);
    return params;
}

static int crf_train_train(
    crf_trainer_t* self,
    const crf_data_t *data,
    const char *filename,
    int holdout
    )
{
    char *algorithm = NULL;
    crf_train_internal_t *tr = (crf_train_internal_t*)self->internal;
    logging_t *lg = tr->lg;
    encoder_t *gm = tr->gm;
    floatval_t *w = NULL;
    dataset_t trainset;
    dataset_t testset;

    /* Prepare the data set(s) for training (and holdout evaluation). */
    dataset_init_trainset(&trainset, (crf_data_t*)data, holdout);
    if (0 <= holdout) {
        dataset_init_testset(&testset, (crf_data_t*)data, holdout);
        logging(lg, "Holdout group: %d\n", holdout+1);
        logging(lg, "\n");
    }

    /* Set the training set to the CRF, and generate features. */
    gm->initialize(gm, &trainset, lg);

    /* Call the training algorithm. */
    switch (tr->algorithm) {
    case TRAIN_LBFGS:
        crf_train_lbfgs(
            gm,
            &trainset,
            (holdout != -1 ? &testset : NULL),
            tr->params,
            lg,
            &w
            );
        break;
    case TRAIN_L2SGD:
        crf_train_l2sgd(
            gm,
            &trainset,
            (holdout != -1 ? &testset : NULL),
            tr->params,
            lg,
            &w
            );
        break;
    case TRAIN_AVERAGED_PERCEPTRON:
        crf_train_averaged_perceptron(
            gm,
            &trainset,
            (holdout != -1 ? &testset : NULL),
            tr->params,
            lg,
            &w
            );
        break;
    case TRAIN_PASSIVE_AGGRESSIVE:
        crf_train_passive_aggressive(
            gm,
            &trainset,
            (holdout != -1 ? &testset : NULL),
            tr->params,
            lg,
            &w
            );
    case TRAIN_AROW:
        crf_train_arow(
            gm,
            &trainset,
            (holdout != -1 ? &testset : NULL),
            tr->params,
            lg,
            &w
            );
        break;
    }

    /* Store the model file. */
    if (filename != NULL && *filename != '\0') {
        gm->save_model(gm, filename, w, lg);
    }

    free(w);

    return 0;
}

int crf1de_create_instance(const char *interface, void **ptr)
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
    } else if (strcmp(interface, "l2sgd") == 0) {
        algorithm = TRAIN_L2SGD;
    } else if (strcmp(interface, "averaged-perceptron") == 0) {
        algorithm = TRAIN_AVERAGED_PERCEPTRON;
    } else if (strcmp(interface, "passive-aggressive") == 0) {
        algorithm = TRAIN_PASSIVE_AGGRESSIVE;
    } else if (strcmp(interface, "arow") == 0) {
        algorithm = TRAIN_AROW;
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
                trainer->train = crf_train_train;

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

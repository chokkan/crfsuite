/*
 *      CRFsuite internal interface.
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

#ifndef __CRFSUITE_INTERNAL_H__
#define __CRFSUITE_INTERNAL_H__

#include <crfsuite.h>
#include "logging.h"

enum {
    FTYPE_NONE = 0,             /**< Unselected. */
    FTYPE_CRF1D,                /**< 1st-order tyad features. */
    FTYPE_CRF1T,                /**< 1st-order triad features. */
};

enum {
    TRAIN_NONE = 0,             /**< Unselected. */
    TRAIN_LBFGS,                /**< L-BFGS batch training. */
    TRAIN_L2SGD,                /**< Pegasos online training. */
    TRAIN_AVERAGED_PERCEPTRON,  /**< Averaged perceptron. */
};

struct tag_crf_train_internal;
typedef struct tag_crf_train_internal crf_train_internal_t;

struct tag_graphical_model;
typedef struct tag_graphical_model graphical_model_t;

typedef struct {
    crf_data_t *data;
    int *perm;
    int num_instances;
} dataset_t;

void dataset_init_trainset(dataset_t *ds, crf_data_t *data, int holdout);
void dataset_init_testset(dataset_t *ds, crf_data_t *data, int holdout);
void dataset_finish(dataset_t *ds);
void dataset_shuffle(dataset_t *ds);
crf_instance_t *dataset_get(dataset_t *ds, int i);

typedef void (*crf_train_enum_features_callback)(void *instance, int fid, floatval_t value);

struct tag_crf_train_internal {
    graphical_model_t *data;       /**< Batch training interface. */
    crf_params_t *params;           /**< Parameter interface. */
    logging_t* lg;                  /**< Logging interface. */
    int feature_type;               /**< Feature type. */
    int algorithm;                  /**< Training algorithm. */
    void *cbe_instance;
};

/**
 * Interface for a graphical model.
 */
struct tag_graphical_model
{
    void *internal;

    const floatval_t *w;
    dataset_t *ds;

    int num_features;
    int cap_items;

    int (*exchange_options)(graphical_model_t *self, crf_params_t* params, int mode);
    int (*set_data)(graphical_model_t *self, dataset_t *ds, logging_t *lg);
    int (*set_weights)(graphical_model_t *self, const floatval_t *w);
    int (*objective)(graphical_model_t *self, const crf_instance_t *inst, floatval_t *f);
    int (*objective_and_gradients)(graphical_model_t *self, const crf_instance_t *inst, floatval_t *f, floatval_t *g, floatval_t scale, floatval_t gain);
    int (*objective_and_gradients_batch)(graphical_model_t *self, dataset_t *ds, floatval_t *f, floatval_t *g);

    int (*tag)(graphical_model_t *self, const crf_instance_t *inst, int *viterbi, floatval_t *ptr_score);
    int (*enum_features)(graphical_model_t *self, const crf_instance_t *seq, const int *labels, crf_train_enum_features_callback func, void *instance);
    int (*save_model)(graphical_model_t *self, const char *filename, const floatval_t *w, logging_t *lg);
};

int crf_train_lbfgs(
    graphical_model_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    );

void crf_train_lbfgs_init(crf_params_t* params);

void crf_train_averaged_perceptron_init(crf_params_t* params);

int crf_train_averaged_perceptron(
    graphical_model_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    );

void crf_train_l2sgd_init(crf_params_t* params);

int crf_train_l2sgd(
    graphical_model_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    );

void holdout_evaluation(
    graphical_model_t *gm,
    dataset_t *testset,
    const floatval_t *w,
    logging_t *lg
    );
    

#endif/*__CRFSUITE_INTERNAL_H__*/

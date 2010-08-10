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

struct tag_crf_train_batch;
typedef struct tag_crf_train_batch crf_train_batch_t;

struct tag_crf_train_internal;
typedef struct tag_crf_train_internal crf_train_internal_t;

struct tag_crf_train_internal {
    crf_train_batch_t *batch;       /**< Batch training interface. */
    crf_params_t *params;           /**< Parameter interface. */
    logging_t* lg;                  /**< Logging interface. */
    crf_evaluate_callback cbe_proc;
    void *cbe_instance;
};

/**
 * Interface for batch training algorithms.
 */
struct tag_crf_train_batch
{
    void *internal;

    const crf_sequence_t *seqs;
    int num_instances;
    int num_attributes;
    int num_labels;
    int num_features;

    int (*exchange_options)(crf_train_batch_t *self, crf_params_t* params, int mode);
    int (*set_data)(crf_train_batch_t *self, const crf_sequence_t *seqs, int num_instances, int num_labels, int num_attributes, logging_t *lg);
    int (*objective_and_gradients)(crf_train_batch_t *self, const floatval_t *w, floatval_t *f, floatval_t *g);
    int (*save_model)(crf_train_batch_t *self, const char *filename, const floatval_t *w, crf_dictionary_t* attrs, crf_dictionary_t* labels, logging_t *lg);
    int (*tag)(crf_train_batch_t *self, const floatval_t *w, crf_sequence_t *inst, crf_output_t* output);
};

int crf_train_lbfgs(
    crf_train_batch_t *batch,
    crf_params_t *params,
    logging_t *lg,
    const crf_sequence_t *seqs,
    int num_instances,
    int num_labels,
    int num_attributes
    );

void crf_train_lbfgs_init(crf_params_t* params);


#endif/*__CRFSUITE_INTERNAL_H__*/

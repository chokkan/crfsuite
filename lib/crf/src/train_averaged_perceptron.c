/*
 *      Online training with averaged perceptron.
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
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crfsuite.h>
#include "crfsuite_internal.h"

#include "logging.h"
#include "params.h"
#include "vecmath.h"

typedef struct {
    int max_iterations;
    floatval_t epsilon;
} option_t;

typedef struct {
    floatval_t *w;
    floatval_t *ws;
    floatval_t c;
    floatval_t cs;
} update_data;

static int exchange_options(crf_params_t* params, option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_FLOAT(
            "ap.epsilon", opt->epsilon, 0.,
            "The stopping criterion (the average number of errors)."
            )

        DDX_PARAM_INT(
            "ap.max_iterations", opt->max_iterations, 10,
            "The maximum number of iterations."
            )

    END_PARAM_MAP()

    return 0;
}

static void update_weights(void *instance, int fid, floatval_t value)
{
    update_data *ud = (update_data*)instance;
    ud->w[fid] += ud->c * value;
    ud->ws[fid] += ud->cs * value;
}

void crf_train_averaged_perceptron_init(crf_params_t* params)
{
    exchange_options(params, NULL, 0);
}

int diff(int *x, int *y, int n)
{
    int i, d = 0;
    for (i = 0;i < n;++i) {
        if (x[i] != y[i]) {
            ++d;
        }
    }
    return d;
}

int crf_train_averaged_perceptron(
    crf_train_batch_t *batch,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    )
{
    int n, i, c, ret;
    int *viterbi = NULL;
    floatval_t *w = NULL;
    floatval_t *ws = NULL;
    floatval_t *wa = NULL;
    clock_t begin = clock();
    const int N = batch->num_instances;
    const int L = batch->num_labels;
    const int A = batch->num_attributes;
    const int K = batch->num_features;
    const int T = batch->max_items;
    const crf_sequence_t *seqs = batch->seqs;
    option_t opt;
    update_data ud;

    exchange_options(params, &opt, -1);

    /* Allocate an array that stores the current weights. */ 
    w = (floatval_t*)calloc(sizeof(floatval_t), K);
    if (w == NULL) {
        return CRFERR_OUTOFMEMORY;
    }

    ws = (floatval_t*)calloc(sizeof(floatval_t), K);
    if (ws == NULL) {
        return CRFERR_OUTOFMEMORY;
    }

    wa = (floatval_t*)calloc(sizeof(floatval_t), K);
    if (wa == NULL) {
        return CRFERR_OUTOFMEMORY;
    }

    viterbi = (int*)calloc(sizeof(int), T);
    if (viterbi == NULL) {
        return CRFERR_OUTOFMEMORY;
    }

    logging(lg, "Averaged perceptron\n");
    logging(lg, "\n");

    ud.w = w;
    ud.ws = ws;

    c = 1;
    for (i = 0;i < opt.max_iterations;++i) {
        floatval_t norm = 0., loss = 0.;
        clock_t iteration_begin = clock();

        for (n = 0;n < N;++n) {
            int d = 0;
            floatval_t score;
            const crf_sequence_t *seq = &seqs[n];

            batch->tag(batch, w, seq, viterbi, &score);
            d = diff(seq->labels, viterbi, seq->num_items);
            if (d != 0) {
                ud.c = 1;
                ud.cs = c;
                batch->enum_features(batch, seq, seq->labels, update_weights, &ud);

                ud.c = -1;
                ud.cs = -c;
                batch->enum_features(batch, seq, viterbi, update_weights, &ud);

                loss += d / (floatval_t)seq->num_items;
            }

            ++c;
        }

        /* Perform averaging to wa. */
        veccopy(wa, w, K);
        vecasub(wa, 1./c, ws, K);

        /* Output the progress. */
        logging(lg, "***** Iteration #%d *****\n", i+1);
        logging(lg, "Loss: %f\n", loss);
        logging(lg, "Feature norm: %f\n", sqrt(vecdot(wa, wa, K)));
        logging(lg, "Seconds required for this iteration: %.3f\n", (clock() - iteration_begin) / (double)CLOCKS_PER_SEC);
        logging(lg, "\n");

        if (loss / N < opt.epsilon) {
            logging(lg, "Terminated with the stopping criterion\n");
            logging(lg, "\n");
            break;
        }
    }

    logging(lg, "Total seconds required for training: %.3f\n", (clock() - begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    *ptr_w = w;
    return 0;
}

/*
 *      Online training with Passive Aggressive.
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
#include <time.h>

#include <crfsuite.h>
#include "crfsuite_internal.h"
#include "logging.h"
#include "params.h"
#include "vecmath.h"

#define MIN(a, b)   ((a) < (b) ? (a) : (b))

/**
 * Training parameters (configurable with crf_params_t interface).
 */
typedef struct {
    int type;
    floatval_t c;
    int error_sensitive;
    int max_iterations;
    floatval_t epsilon;
} training_option_t;

/**
 * Internal data structure for computing F(x, y) - F(x, y').
 */
typedef struct {
    /* An array of feature indices relevant to the instance. */
    int *actives;
    int num_actives;
    int cap_actives;

    /* The number of features. */
    int K;
    floatval_t c;
    /* The difference vector [K]. */
    floatval_t *delta;
    /* Flags [K] to indicate whether delta[i] is used. */
    char *used;
} delta_t;

static int delta_init(delta_t *dc, const int K)
{
    memset(dc, 0, sizeof(*dc));
    dc->K = K;
    dc->delta = (floatval_t*)calloc(K, sizeof(floatval_t));
    dc->used = (char*)calloc(K, sizeof(char));
    if (dc->delta == NULL || dc->used == NULL) {
        return 1;
    }
    return 0;
}

static void delta_finish(delta_t *dc)
{
    free(dc->actives);
    free(dc->used);
    free(dc->delta);
    memset(dc, 0, sizeof(*dc));
}

static void delta_reset(delta_t *dc)
{
    int i;
    for (i = 0;i < dc->num_actives;++i) {
        int k = dc->actives[i];
        dc->delta[k] = 0;
    }
    dc->num_actives = 0;
}

static void delta_reset_used(delta_t *dc)
{
    int i;
    for (i = 0;i < dc->num_actives;++i) {
        int k = dc->actives[i];
        dc->used[k] = 0;
    }
}

static floatval_t delta_norm2(delta_t *dc)
{
    int i;
    floatval_t norm2 = 0.;

    for (i = 0;i < dc->num_actives;++i) {
        int k = dc->actives[i];
        if (!dc->used[k]) {
            norm2 += dc->delta[k] * dc->delta[k];
            dc->used[k] = 1;
        }
    }
    delta_reset_used(dc);
    return norm2;
}

static void delta_add(delta_t *dc, floatval_t *w, floatval_t tau)
{
    int i;
    memset(dc->used, 0, sizeof(char) * dc->K);
    for (i = 0;i < dc->num_actives;++i) {
        int k = dc->actives[i];
        if (!dc->used[k]) {
            w[k] += tau * dc->delta[k];
            dc->used[k] = 1;
        }
    }
    delta_reset_used(dc);
}

static void delta_collect(void *instance, int fid, floatval_t value)
{
    delta_t *dc = (delta_t*)instance;

    /* Expand the active feature list if necessary. */
    if (dc->cap_actives <= dc->num_actives) {
        ++dc->cap_actives;
        dc->cap_actives *= 2;
        dc->actives = (int*)realloc(dc->actives, sizeof(int) * dc->cap_actives);
    }

    dc->actives[dc->num_actives++] = fid;
    dc->delta[fid] += dc->c * value;
}

static int diff(int *x, int *y, int n)
{
    int i, d = 0;
    for (i = 0;i < n;++i) {
        if (x[i] != y[i]) {
            ++d;
        }
    }
    return d;
}

static floatval_t cost_insensitive(floatval_t err, floatval_t d)
{
    return err + 1.;
}

static floatval_t cost_sensitive(floatval_t err, floatval_t d)
{
    return err + sqrt(d);
}

static floatval_t tau0(floatval_t cost, floatval_t norm, floatval_t c)
{
    return cost / norm;
}

static floatval_t tau1(floatval_t cost, floatval_t norm, floatval_t c)
{
    return MIN(c, cost / norm);
}

static floatval_t tau2(floatval_t cost, floatval_t norm, floatval_t c)
{
    return cost / (norm + 0.5 / c);
}

static int exchange_options(crf_params_t* params, training_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_INT(
            "type", opt->type, 1,
            "The strategy for updating feature weights, {0, 1, 2}."
            )
        DDX_PARAM_FLOAT(
            "c", opt->c, 1.,
            "The aggressiveness parameter."
            )
        DDX_PARAM_INT(
            "error_sensitive", opt->error_sensitive, 1,
            "Cost is sensitive to the number of incorrect labels."
            )
        DDX_PARAM_INT(
            "max_iterations", opt->max_iterations, 100,
            "The maximum number of iterations."
            )
        DDX_PARAM_FLOAT(
            "epsilon", opt->epsilon, 0.,
            "The stopping criterion (the average number of errors)."
            )
    END_PARAM_MAP()

    return 0;
}

void crf_train_passive_aggressive_init(crf_params_t* params)
{
    exchange_options(params, NULL, 0);
}

int crf_train_passive_aggressive(
    encoder_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    )
{
    int n, i, c, ret = 0;
    int *viterbi = NULL;
    floatval_t *w = NULL;
    const int N = trainset->num_instances;
    const int K = gm->num_features;
    const int T = gm->cap_items;
    training_option_t opt;
    delta_t dc;
    clock_t begin = clock();
    floatval_t (*cost_function)(floatval_t err, floatval_t d) = NULL;
    floatval_t (*tau_function)(floatval_t cost, floatval_t norm, floatval_t c) = NULL;

	/* Initialize the variable. */
    if (delta_init(&dc, K) != 0) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Obtain parameter values. */
    exchange_options(params, &opt, -1);

    /* Allocate arrays. */
    w = (floatval_t*)calloc(sizeof(floatval_t), K);
    viterbi = (int*)calloc(sizeof(int), T);
    if (w == NULL || viterbi == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    if (opt.error_sensitive) {
        cost_function = cost_sensitive;
    } else {
        cost_function = cost_insensitive;
    }

    if (opt.type == 1) {
        tau_function = tau1;
    } else if (opt.type == 2) {
        tau_function = tau2;
    } else {
        tau_function = tau0;
    }

    /* Show the parameters. */
    logging(lg, "Passive Aggressive\n");
    logging(lg, "type: %d\n", opt.type);
    logging(lg, "c: %f\n", opt.c);
    logging(lg, "error_sensitive: %d\n", opt.error_sensitive);
    logging(lg, "max_iterations: %d\n", opt.max_iterations);
    logging(lg, "epsilon: %f\n", opt.epsilon);
    logging(lg, "\n");

	/* Loop for epoch. */
    for (i = 0;i < opt.max_iterations;++i) {
        floatval_t norm = 0., sum_loss = 0.;
        clock_t iteration_begin = clock();

        /* Shuffle the instances. */
        dataset_shuffle(trainset);

		/* Loop for each instance. */
        for (n = 0;n < N;++n) {
            int d = 0;
            floatval_t sv;
            const crf_instance_t *inst = dataset_get(trainset, n);

            /* Set the feature weights to the graphical model. */
            gm->set_weights(gm, w, 1.);
            gm->set_instance(gm, inst);

            /* Tag the sequence with the current model. */
            gm->viterbi(gm, viterbi, &sv);

            /* Compute the number of different labels. */
            d = diff(inst->labels, viterbi, inst->num_items);
            if (0 < d) {
                floatval_t sc, norm2;
                floatval_t tau, cost;

                /* Compute the loss. */
                gm->score(gm, inst->labels, &sc);
                cost = cost_function(sv - sc, (double)d);

                delta_reset(&dc);

                /*
                    For every feature k on the correct path:
                        delta[k] += 1;
                 */
                dc.c = 1;
                gm->features_on_path(gm, inst, inst->labels, delta_collect, &dc);

                /*
                    For every feature k on the Viterbi path:
                        delta[k] -= 1;
                 */
                dc.c = -1;
                gm->features_on_path(gm, inst, viterbi, delta_collect, &dc);

                /* Compute the ||delta||^2. */
                norm2 = delta_norm2(&dc);
                tau = tau_function(cost, norm2, opt.c);
                delta_add(&dc, w, tau);

                sum_loss += cost;
            }
        }

        /* Output the progress. */
        logging(lg, "***** Iteration #%d *****\n", i+1);
        logging(lg, "Loss: %f\n", sum_loss);
        logging(lg, "Feature norm: %f\n", sqrt(vecdot(w, w, K)));
        logging(lg, "Seconds required for this iteration: %.3f\n", (clock() - iteration_begin) / (double)CLOCKS_PER_SEC);

        /* Holdout evaluation if necessary. */
        if (testset != NULL) {
            holdout_evaluation(gm, testset, w, lg);
        }

        logging(lg, "\n");

        /* Convergence test. */
        if (sum_loss / N < opt.epsilon) {
            logging(lg, "Terminated with the stopping criterion\n");
            logging(lg, "\n");
            break;
        }
    }

    logging(lg, "Total seconds required for training: %.3f\n", (clock() - begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    free(viterbi);
    *ptr_w = w;
    delta_finish(&dc);
    return ret;

error_exit:
    free(viterbi);
    free(w);
    delta_finish(&dc);
    *ptr_w = NULL;

    return ret;
}

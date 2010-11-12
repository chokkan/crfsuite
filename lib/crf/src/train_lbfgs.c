/*
 *      Batch training with L-BFGS method.
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
#include <lbfgs.h>

typedef struct {
    char*       regularization;
    floatval_t  regularization_sigma;
    int         memory;
    floatval_t  epsilon;
    int         stop;
    floatval_t  delta;
    int         max_iterations;
    char*       linesearch;
    int         linesearch_max_iterations;
} crf1dl_lbfgs_option_t;

typedef struct {
    crf_train_batch_t *batch;
    logging_t *lg;
    int l2_regularization;
    floatval_t sigma2inv;
    floatval_t* best_w;
    clock_t begin;
    crf_evaluate_callback cbe_proc;
    void *cbe_instance;
} lbfgs_internal_t;

static lbfgsfloatval_t lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    floatval_t f, norm = 0.;
    lbfgs_internal_t *lbfgsi = (lbfgs_internal_t*)instance;
    crf_train_batch_t *batch = lbfgsi->batch;

    /* Compute the objective value and gradients. */
    batch->objective_and_gradients(batch, x, &f, g);
    
    /* L2 regularization. */
    if (lbfgsi->l2_regularization) {
        for (i = 0;i < n;++i) {
            g[i] += (lbfgsi->sigma2inv * x[i]);
            norm += x[i] * x[i];
        }
        f += (lbfgsi->sigma2inv * norm * 0.5);
    }

    return f;
}

static int lbfgs_progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls)
{
    int i, num_active_features = 0;
    clock_t duration, clk = clock();
    lbfgs_internal_t *lbfgsi = (lbfgs_internal_t*)instance;
    crf_train_batch_t *batch = lbfgsi->batch;
    logging_t *lg = lbfgsi->lg;

    /* Compute the duration required for this iteration. */
    duration = clk = lbfgsi->begin;
    lbfgsi->begin = clk;

    /* Set feature weights from the L-BFGS solver. */
    for (i = 0;i < n;++i) {
        lbfgsi->best_w[i] = x[i];
        if (x[i] != 0.) ++num_active_features;
    }

    /* Report the progress. */
    logging(lg, "***** Iteration #%d *****\n", k);
    logging(lg, "Log-likelihood: %f\n", -fx);
    logging(lg, "Feature norm: %f\n", xnorm);
    logging(lg, "Error norm: %f\n", gnorm);
    logging(lg, "Active features: %d\n", num_active_features);
    logging(lg, "Line search trials: %d\n", ls);
    logging(lg, "Line search step: %f\n", step);
    logging(lg, "Seconds required for this iteration: %.3f\n", duration / (double)CLOCKS_PER_SEC);

    /* Send the tagger with the current parameters. */
#if 0
    if (crf1dt->cbe_proc != NULL) {
        /* Callback notification with the tagger object. */
        int ret = crf1dt->cbe_proc(crf1dt->cbe_instance, &crf1dt->tagger);
    }
#endif

    logging(lg, "\n");

    /* Continue. */
    return 0;
}

int crf1dl_lbfgs_options(crf_params_t* params, crf1dl_lbfgs_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_STRING(
            "regularization", opt->regularization, "L2",
            "Specify the regularization type."
            )
        DDX_PARAM_FLOAT(
            "regularization.sigma", opt->regularization_sigma, 10.0,
            "Specify the regularization constant."
            )
        DDX_PARAM_INT(
            "lbfgs.max_iterations", opt->max_iterations, INT_MAX,
            "The maximum number of L-BFGS iterations."
            )
        DDX_PARAM_INT(
            "lbfgs.num_memories", opt->memory, 6,
            "The number of corrections to approximate the inverse hessian matrix."
            )
        DDX_PARAM_FLOAT(
            "lbfgs.epsilon", opt->epsilon, 1e-5,
            "Epsilon for testing the convergence of the objective."
            )
        DDX_PARAM_INT(
            "lbfgs.stop", opt->stop, 10,
            "The duration of iterations to test the stopping criterion."
            )
        DDX_PARAM_FLOAT(
            "lbfgs.delta", opt->delta, 1e-5,
            "The threshold for the stopping criterion; an L-BFGS iteration stops when the\n"
            "improvement of the log likelihood over the last ${lbfgs.stop} iterations is\n"
            "no greater than this threshold."
            )
        DDX_PARAM_STRING(
            "lbfgs.linesearch", opt->linesearch, "MoreThuente",
            "The line search algorithm used in L-BFGS updates:\n"
            "{'MoreThuente': More and Thuente's method, 'Backtracking': backtracking}"
            )
        DDX_PARAM_INT(
            "lbfgs.linesearch.max_iterations", opt->linesearch_max_iterations, 20,
            "The maximum number of trials for the line search algorithm."
            )
    END_PARAM_MAP()

    return 0;
}


void crf_train_lbfgs_init(crf_params_t* params)
{
    crf1dl_lbfgs_options(params, NULL, 0);
}

int crf_train_lbfgs(
    crf_train_batch_t *batch,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w,
    crf_evaluate_callback cbe_proc,
    void *cbe_instance
    )
{
    int i, ret;
    floatval_t *w = NULL;
    clock_t begin = clock();
    const int N = batch->num_instances;
    const int L = batch->num_labels;
    const int A = batch->num_attributes;
    const int K = batch->num_features;
    lbfgs_internal_t lbfgsi;
    lbfgs_parameter_t lbfgsparam;
    crf1dl_lbfgs_option_t opt;

    /* Allocate an array that stores the current weights. */ 
    w = (floatval_t*)calloc(sizeof(floatval_t), K);
    if (w == NULL) {
        return CRFERR_OUTOFMEMORY;
    }
 
    /* Allocate an array that stores the best weights. */ 
    lbfgsi.best_w = (floatval_t*)calloc(sizeof(floatval_t), K);
    if (lbfgsi.best_w == NULL) {
        return CRFERR_OUTOFMEMORY;
    }

    /* Read the L-BFGS parameters. */
    crf1dl_lbfgs_options(params, &opt, -1);
    logging(lg, "L-BFGS optimization\n");
    logging(lg, "regularization: %s\n", opt.regularization);
    logging(lg, "regularization.sigma: %f\n", opt.regularization_sigma);
    logging(lg, "lbfgs.num_memories: %d\n", opt.memory);
    logging(lg, "lbfgs.max_iterations: %d\n", opt.max_iterations);
    logging(lg, "lbfgs.epsilon: %f\n", opt.epsilon);
    logging(lg, "lbfgs.stop: %d\n", opt.stop);
    logging(lg, "lbfgs.delta: %f\n", opt.delta);
    logging(lg, "lbfgs.linesearch: %s\n", opt.linesearch);
    logging(lg, "lbfgs.linesearch.max_iterations: %d\n", opt.linesearch_max_iterations);
    logging(lg, "\n");

    /* Set parameters for L-BFGS. */
    lbfgs_parameter_init(&lbfgsparam);
    lbfgsparam.m = opt.memory;
    lbfgsparam.epsilon = opt.epsilon;
    lbfgsparam.past = opt.stop;
    lbfgsparam.delta = opt.delta;
    lbfgsparam.max_iterations = opt.max_iterations;
    if (strcmp(opt.linesearch, "Backtracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(opt.linesearch, "StrongBacktracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    } else {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    }
    lbfgsparam.max_linesearch = opt.linesearch_max_iterations;

    /* Set regularization parameters. */
    if (strcmp(opt.regularization, "L1") == 0) {
        lbfgsi.l2_regularization = 0;
        lbfgsparam.orthantwise_c = 1.0 / opt.regularization_sigma;
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(opt.regularization, "L2") == 0) {
        lbfgsi.l2_regularization = 1;
        lbfgsi.sigma2inv = 1.0 / (opt.regularization_sigma * opt.regularization_sigma);
        lbfgsparam.orthantwise_c = 0.;
    } else {
        lbfgsi.l2_regularization = 0;
        lbfgsparam.orthantwise_c = 0.;
    }

    /* Set other callback data. */
    lbfgsi.batch = batch;
    lbfgsi.lg = lg;
    lbfgsi.cbe_proc = cbe_proc;
    lbfgsi.cbe_instance = cbe_instance;

    /* Call the L-BFGS solver. */
    lbfgsi.begin = clock();
    ret = lbfgs(
        K,
        w,
        NULL,
        lbfgs_evaluate,
        lbfgs_progress,
        &lbfgsi,
        &lbfgsparam
        );
    if (ret == LBFGS_CONVERGENCE) {
        logging(lg, "L-BFGS resulted in convergence\n");
    } else if (ret == LBFGS_STOP) {
        logging(lg, "L-BFGS terminated with the stopping criteria\n");
    } else if (ret == LBFGSERR_MAXIMUMITERATION) {
        logging(lg, "L-BFGS terminated with the maximum number of iterations\n");
    } else {
        logging(lg, "L-BFGS terminated with error code (%d)\n", ret);
    }


    for (i = 0;i < K;++i) {
        w[i] = lbfgsi.best_w[i];
    }

    logging(lg, "Total seconds required for training: %.3f\n", (clock() - begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    *ptr_w = w;
    return 0;
}

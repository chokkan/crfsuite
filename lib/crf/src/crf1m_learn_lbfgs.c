/*
 *      Training linear-chain CRF with L-BFGS.
 *
 * Copyright (c) 2007,2008, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Northwestern University, University of Tokyo,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written
 *       permission.
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

#ifdef	HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crf.h>
#include "crf1m.h"

#include "logging.h"
#include "params.h"
#include <lbfgs.h>

typedef struct {
    floatval_t* g;
} lbfgs_internal_t;

#define LBFGS_INTERNAL(crf1ml)    ((lbfgs_internal_t*)((crf1ml)->solver_data))

inline static void update_model_expectations(
    crf1ml_feature_t* f,
    const int fid,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* crf1ml,
    const crf_sequence_t* seq,
    int t
    )
{
    lbfgs_internal_t *lbfgsi = LBFGS_INTERNAL(crf1ml);
    lbfgsi->g[fid] += prob * scale;
}

static lbfgsfloatval_t lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
	int i;
	floatval_t logp = 0, logl = 0, norm = 0;
	crf1ml_t* crf1mt = (crf1ml_t*)instance;
	crf_sequence_t* seqs = crf1mt->seqs;
	const int N = crf1mt->num_sequences;
    lbfgs_internal_t *lbfgsi = LBFGS_INTERNAL(crf1mt);

    /* Set the gradient vector. */
    lbfgsi->g = g;

	/*
		Set feature weights from the L-BFGS solver. Initialize model
		expectations as zero.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1ml_feature_t* f = &crf1mt->features[i];
        g[i] = -f->freq;
	}

	/*
		Set the scores (weights) of transition features here because
		these are independent of input label sequences.
	 */
	crf1ml_transition_score(crf1mt, x, n, 1.0);
	crf1mc_exp_transition(crf1mt->ctx);

	/*
		Compute model expectations.
	 */
	for (i = 0;i < N;++i) {
		/* Set label sequences and state scores. */
		crf1ml_set_labels(crf1mt, &seqs[i]);
		crf1ml_state_score(crf1mt, &seqs[i], x, n, 1.0);
		crf1mc_exp_state(crf1mt->ctx);

		/* Compute forward/backward scores. */
		crf1mc_forward_score(crf1mt->ctx);
		crf1mc_backward_score(crf1mt->ctx);

		/*crf1mc_debug_context(crf1mt->ctx, stdout);*/
		/*printf("lognorm = %f\n", crf1mt->ctx->log_norm);*/

		/* Compute the probability of the input sequence on the model. */
		logp = crf1mc_logprob(crf1mt->ctx);
		/* Update the log-likelihood. */
		logl += logp;

		/* Update the model expectations of features. */
		crf1ml_enum_features(crf1mt, &seqs[i], update_model_expectations);
	}

	/*
		L2 regularization.
		Note that we *add* the (w * sigma) to g[i].
	 */
	if (crf1mt->l2_regularization) {
		for (i = 0;i < crf1mt->num_features;++i) {
			const crf1ml_feature_t* f = &crf1mt->features[i];
			g[i] += (crf1mt->sigma2inv * x[i]);
			norm += x[i] * x[i];
		}
		logl -= (crf1mt->sigma2inv * norm * 0.5);
	}

	return -logl;
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
	crf1ml_t* crf1mt = (crf1ml_t*)instance;

	/* Compute the duration required for this iteration. */
	duration = clk - crf1mt->clk_prev;
	crf1mt->clk_prev = clk;

	/* Set feature weights from the L-BFGS solver. */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt->best_w[i] = x[i];
		if (x[i] != 0.) ++num_active_features;
	}

	/* Report the progress. */
	logging(crf1mt->lg, "***** Iteration #%d *****\n", k);
	logging(crf1mt->lg, "Log-likelihood: %f\n", -fx);
	logging(crf1mt->lg, "Feature norm: %f\n", xnorm);
	logging(crf1mt->lg, "Error norm: %f\n", gnorm);
	logging(crf1mt->lg, "Active features: %d\n", num_active_features);
	logging(crf1mt->lg, "Line search trials: %d\n", ls);
	logging(crf1mt->lg, "Line search step: %f\n", step);
	logging(crf1mt->lg, "Seconds required for this iteration: %.3f\n", duration / (double)CLOCKS_PER_SEC);

	/* Send the tagger with the current parameters. */
	if (crf1mt->cbe_proc != NULL) {
		/* Callback notification with the tagger object. */
		int ret = crf1mt->cbe_proc(crf1mt->cbe_instance, &crf1mt->tagger);
	}
	logging(crf1mt->lg, "\n");

	/* Continue. */
	return 0;
}

int crf1ml_lbfgs_options(crf_params_t* params, crf1ml_option_t* opt, int mode)
{
    crf1ml_lbfgs_option_t* lbfgs = &opt->lbfgs;

	BEGIN_PARAM_MAP(params, mode)
		DDX_PARAM_INT("lbfgs.max_iterations", lbfgs->max_iterations, INT_MAX)
		DDX_PARAM_INT("lbfgs.num_memories", lbfgs->memory, 6)
		DDX_PARAM_FLOAT("lbfgs.epsilon", lbfgs->epsilon, 1e-5)
		DDX_PARAM_INT("lbfgs.stop", lbfgs->stop, 10)
		DDX_PARAM_FLOAT("lbfgs.delta", lbfgs->delta, 1e-5)
		DDX_PARAM_STRING("lbfgs.linesearch", lbfgs->linesearch, "MoreThuente")
		DDX_PARAM_INT("lbfgs.linesearch.max_iterations", lbfgs->linesearch_max_iterations, 20)
	END_PARAM_MAP()

	return 0;
}


int crf1ml_lbfgs(
    crf1ml_t* crf1mt,
    crf1ml_option_t *opt
    )
{
    int ret;
    lbfgs_internal_t lbfgsi;
	lbfgs_parameter_t lbfgsparam;
    crf1ml_lbfgs_option_t* lbfgsopt = &opt->lbfgs;

    crf1mt->solver_data = &lbfgsi;

	/* Initialize the L-BFGS parameters with default values. */
	lbfgs_parameter_init(&lbfgsparam);

	logging(crf1mt->lg, "L-BFGS optimization\n");
	logging(crf1mt->lg, "lbfgs.num_memories: %d\n", lbfgsopt->memory);
	logging(crf1mt->lg, "lbfgs.max_iterations: %d\n", lbfgsopt->max_iterations);
	logging(crf1mt->lg, "lbfgs.epsilon: %f\n", lbfgsopt->epsilon);
	logging(crf1mt->lg, "lbfgs.stop: %d\n", lbfgsopt->stop);
	logging(crf1mt->lg, "lbfgs.delta: %f\n", lbfgsopt->delta);
	logging(crf1mt->lg, "lbfgs.linesearch: %s\n", lbfgsopt->linesearch);
	logging(crf1mt->lg, "lbfgs.linesearch.max_iterations: %d\n", lbfgsopt->linesearch_max_iterations);
	logging(crf1mt->lg, "\n");

	/* Set parameters for L-BFGS. */
	lbfgsparam.m = lbfgsopt->memory;
	lbfgsparam.epsilon = lbfgsopt->epsilon;
    lbfgsparam.past = lbfgsopt->stop;
    lbfgsparam.delta = lbfgsopt->delta;
	lbfgsparam.max_iterations = lbfgsopt->max_iterations;
    if (strcmp(lbfgsopt->linesearch, "Backtracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(lbfgsopt->linesearch, "StrongBacktracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG;
    } else {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    }
    lbfgsparam.max_linesearch = lbfgsopt->linesearch_max_iterations;

	/* Set regularization parameters. */
	if (strcmp(opt->regularization, "L1") == 0) {
		crf1mt->l2_regularization = 0;
		lbfgsparam.orthantwise_c = 1.0 / opt->regularization_sigma;
	} else if (strcmp(opt->regularization, "L2") == 0) {
		crf1mt->l2_regularization = 1;
		crf1mt->sigma2inv = 1.0 / (opt->regularization_sigma * opt->regularization_sigma);
		lbfgsparam.orthantwise_c = 0.;
    } else {
        crf1mt->l2_regularization = 0;
        lbfgsparam.orthantwise_c = 0.;
    }

	/* Call the L-BFGS solver. */
	crf1mt->clk_begin = clock();
	crf1mt->clk_prev = crf1mt->clk_begin;
	ret = lbfgs(
		crf1mt->num_features,
		crf1mt->w,
		NULL,
		lbfgs_evaluate,
		lbfgs_progress,
		crf1mt,
		&lbfgsparam
		);
    if (ret == LBFGS_CONVERGENCE) {
		logging(crf1mt->lg, "L-BFGS resulted in convergence\n");
    } else if (ret == LBFGS_STOP) {
		logging(crf1mt->lg, "L-BFGS terminated with the stopping criteria\n");
	} else if (ret == LBFGSERR_MAXIMUMITERATION) {
		logging(crf1mt->lg, "L-BFGS terminated with the maximum number of iterations\n");
	} else {
		logging(crf1mt->lg, "L-BFGS terminated with error code (%d)\n", ret);
	}
	logging(crf1mt->lg, "Total seconds required for L-BFGS: %.3f\n", (clock() - crf1mt->clk_begin) / (double)CLOCKS_PER_SEC);
	logging(crf1mt->lg, "\n");

    return 0;
}

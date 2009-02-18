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
#include <time.h>

#include <crf.h>

#include "logging.h"
#include "crf1m.h"
#include <lbfgs.h>

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

	/*
		Set feature weights from the L-BFGS solver. Initialize model
		expectations as zero.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1ml_feature_t* f = &crf1mt->features[i];
		f->lambda = x[i];
		f->mexp = 0;
        f->oexp = 0;
	}

	/*
		Set the scores (weights) of transition features here because
		these are independent of input label sequences.
	 */
	crf1ml_transition_score(crf1mt);
	crf1mc_exp_transition(crf1mt->ctx);

	/*
		Compute model expectations.
	 */
	for (i = 0;i < N;++i) {
		/* Set label sequences and state scores. */
		crf1ml_set_labels(crf1mt, &seqs[i]);
		crf1ml_state_score(crf1mt, &seqs[i]);
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
		crf1ml_accumulate_expectation(crf1mt, &seqs[i]);
	}

	/*
		Update the gradient vector.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		const crf1ml_feature_t* f = &crf1mt->features[i];
		g[i] = -(f->oexp - f->mexp);
	}

	/*
		L2 regularization.
		Note that we *add* the (lambda * sigma) to g[i].
	 */
	if (crf1mt->l2_regularization) {
		for (i = 0;i < crf1mt->num_features;++i) {
			const crf1ml_feature_t* f = &crf1mt->features[i];
			g[i] += (crf1mt->sigma2inv * f->lambda);
			norm += f->lambda * f->lambda;
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
		crf1ml_feature_t* f = &crf1mt->features[i];
		f->lambda = x[i];
		crf1mt->best_lambda[i] = x[i];
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

int crf1ml_lbfgs(
    crf1ml_t* crf1mt,
    crf1ml_option_t *opt
    )
{
    int ret;
	lbfgs_parameter_t lbfgsopt;

	/* Initialize the L-BFGS parameters with default values. */
	lbfgs_parameter_init(&lbfgsopt);

	logging(crf1mt->lg, "L-BFGS optimization\n");
	logging(crf1mt->lg, "lbfgs.num_memories: %d\n", opt->lbfgs_memory);
	logging(crf1mt->lg, "lbfgs.max_iterations: %d\n", opt->lbfgs_max_iterations);
	logging(crf1mt->lg, "lbfgs.epsilon: %f\n", opt->lbfgs_epsilon);
	logging(crf1mt->lg, "lbfgs.stop: %d\n", opt->lbfgs_stop);
	logging(crf1mt->lg, "lbfgs.delta: %f\n", opt->lbfgs_delta);
	logging(crf1mt->lg, "lbfgs.linesearch: %s\n", opt->lbfgs_linesearch);
	logging(crf1mt->lg, "lbfgs.linesearch.max_iterations: %d\n", opt->lbfgs_linesearch_max_iterations);
	logging(crf1mt->lg, "\n");

	/* Set parameters for L-BFGS. */
	lbfgsopt.m = opt->lbfgs_memory;
	lbfgsopt.epsilon = opt->lbfgs_epsilon;
    lbfgsopt.past = opt->lbfgs_stop;
    lbfgsopt.delta = opt->lbfgs_delta;
	lbfgsopt.max_iterations = opt->lbfgs_max_iterations;
    if (strcmp(opt->lbfgs_linesearch, "Backtracking") == 0) {
        lbfgsopt.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(opt->lbfgs_linesearch, "LooseBacktracking") == 0) {
        lbfgsopt.linesearch = LBFGS_LINESEARCH_BACKTRACKING_LOOSE;
    } else {
        lbfgsopt.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    }
    lbfgsopt.max_linesearch = opt->lbfgs_linesearch_max_iterations;

	/* Set regularization parameters. */
	if (strcmp(opt->regularization, "L1") == 0) {
		crf1mt->l2_regularization = 0;
		lbfgsopt.orthantwise_c = 1.0 / opt->regularization_sigma;
	} else if (strcmp(opt->regularization, "L2") == 0) {
		crf1mt->l2_regularization = 1;
		crf1mt->sigma2inv = 1.0 / (opt->regularization_sigma * opt->regularization_sigma);
		lbfgsopt.orthantwise_c = 0.;
    } else {
        crf1mt->l2_regularization = 0;
        lbfgsopt.orthantwise_c = 0.;
    }

	/* Call the L-BFGS solver. */
	crf1mt->clk_begin = clock();
	crf1mt->clk_prev = crf1mt->clk_begin;
	ret = lbfgs(
		crf1mt->num_features,
		crf1mt->lambda,
		NULL,
		lbfgs_evaluate,
		lbfgs_progress,
		crf1mt,
		&lbfgsopt
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

/*
 *      Training linear-chain CRFs with Stochastic Gradient Descent (SGD).
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

/*
    SGD for L2-regularized MAP estimation.

    The iterative algorithm is based on Pegasos:

    Shai Shalev-Shwartz, Yoram Singer, and Nathan Srebro.
    Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
    In Proc. of ICML 2007, pp 807-814, 2007.

    The objective function to minimize:
        
        f(w) = (lambda/2) * ||w||^2 + (1/N) * \sum_i^N log P^i(y|x)

    The Pegasos algorithm.

    0) Initialization
        t = 0
    1) Computing the learning rate (eta).
        eta = 1 / (lambda * t)
    2) Updating the feature weights.
        w = (1 - eta * lambda) w + (eta / k) \sum_i (oexp - mexp)
    3) Projecting the feature weights within L2-ball.
        w = min{1, (1/sqrt(lambda))/||w||} * w
    4) Goto 1 until convergence.

    0)
        decay = 1
        scale = 1
    1)
        eta = 1 / (lambda * t)
        decay *= (1 - eta * lambda)
        gain = (eta / k) / (decay * scale)
    2)
        norm2 -= w * w
        w -= gain * P(y|x) * f(x,y)
        norm2 += w * w
        norm2 -= w * w
        w += gain * f(x, y)
        norm2 += w * w
    3)
        if norm2 * decay^2 * scale^2 * lambda > 1:
            scale = 1 / (sqrt(norm2 * lambda) * decay * scale)

        
*/


#ifdef	HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <crf.h>

#include "logging.h"
#include "crf1m.h"

#define MIN(a, b)   ((a) < (b) ? (a) : (b))

inline void
update_weight(
    crf1ml_t* trainer,
    crf1ml_feature_t* f,
    floatval_t a
    )
{
    floatval_t w = f->w;
    f->w += a;
    trainer->norm += a * (a + w + w);
}

inline static void update_feature_weights(
    crf1ml_feature_t* f,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* trainer,
    const crf_sequence_t* seq,
    int t
    )
{
    // Subtract the model expectation from the weight.
    update_weight(trainer, f, -trainer->gain * prob * scale);

    switch (f->type) {
    case FT_STATE:      /**< State features. */
        if (f->dst == seq->items[t].label) {
            update_weight(trainer, f, trainer->gain * scale);
        }
        break;
    case FT_TRANS:      /**< Transition features. */
        if (f->src == seq->items[t].label &&
            f->dst == seq->items[t+1].label) {
            update_weight(trainer, f, trainer->gain * scale);
        }
        break;
    case FT_TRANS_BOS:  /**< BOS transition features. */
        if (f->dst == seq->items[t].label) {
            update_weight(trainer, f, trainer->gain * scale);
        }
        break;
    case FT_TRANS_EOS:  /**< EOS transition features. */
        if (f->src == seq->items[t].label) {
            update_weight(trainer, f, trainer->gain * scale);
        }
        break;
    }
}

void scale_weights(crf1ml_feature_t* fs, const int n, const floatval_t scale)
{
    int i;
    for (i = 0;i < n;++i) {
        fs[i].w *= scale;
    }
}

static floatval_t
compute_loglikelihood(
    crf1ml_t* crf1mt,
    int *perm,
    const int N,
    floatval_t lambda
    )
{
    int i;
	crf_sequence_t *seq;
    floatval_t logp = 0., norm = 0.;
    const int K = crf1mt->num_features;

    /* Set transition scores. */
    crf1ml_transition_score(crf1mt, 1.);
    crf1mc_exp_transition(crf1mt->ctx);

    for (i = 0;i < N;++i) {
        seq = &crf1mt->seqs[perm[i]];

	    /* Set label sequences and state scores. */
	    crf1ml_set_labels(crf1mt, seq);
	    crf1ml_state_score(crf1mt, seq, 1.);
	    crf1mc_exp_state(crf1mt->ctx);

	    /* Compute forward/backward scores. */
	    crf1mc_forward_score(crf1mt->ctx);
	    crf1mc_backward_score(crf1mt->ctx);

	    /* Compute the probability of the input sequence on the model. */
        logp += crf1mc_logprob(crf1mt->ctx);
    }

    /* Compute the L2 norm of feature weights. */
    for (i = 0;i < K;++i) {
        norm += crf1mt->w[i] * crf1mt->w[i];
    }

    return logp - lambda / 2 * norm;
}



static floatval_t l2sgd(
    crf1ml_t* crf1mt,
    int *perm,
    const int N,
    const floatval_t t0,
    const floatval_t lambda,
    const int num_epochs,
    int calibration
    )
{
    int epoch, i;
    floatval_t t = 0;
    floatval_t logp = 0;
    clock_t clk_prev;
	crf_sequence_t *seq, *seqs = crf1mt->seqs;
    const int K = crf1mt->num_features;
    floatval_t eta, scale, boundary;
    floatval_t decay = 1., proj = 1.;

    /* Loop for epochs. */
    for (epoch = 1;epoch <= num_epochs;++epoch) {
        if (!calibration) {
	        logging(crf1mt->lg, "***** Epoch #%d *****\n", epoch);
        }
        clk_prev = clock();

        if (!calibration) {
            /* Generate a permutation that shuffles the instances. */
            crf1ml_shuffle(perm, N, 0);
        }

        /* Loop for instances. */
        logp = 0.;
        for (i = 0;i < N;++i) {
            seq = &seqs[perm[i]];

            /* Update various factors. */
            eta = 1 / (lambda * (t0 + t));
            decay *= (1.0 - eta * lambda);
            scale = decay * proj;
            crf1mt->gain = eta / scale;

            /* Set transition scores. */
            crf1ml_transition_score(crf1mt, scale);
            crf1mc_exp_transition(crf1mt->ctx);

		    /* Set label sequences and state scores. */
		    crf1ml_set_labels(crf1mt, seq);
		    crf1ml_state_score(crf1mt, seq, scale);
		    crf1mc_exp_state(crf1mt->ctx);

		    /* Compute forward/backward scores. */
		    crf1mc_forward_score(crf1mt->ctx);
		    crf1mc_backward_score(crf1mt->ctx);

		    /* Compute the probability of the input sequence on the model. */
            logp += crf1mc_logprob(crf1mt->ctx);

		    /* Update the feature weights. */
		    crf1ml_enum_features(crf1mt, seq, update_feature_weights);

            /* Project feature weights in the L2-ball. */
            boundary = crf1mt->norm * scale * scale * lambda;
            if (1. < boundary) {
                proj = 1.0 / sqrt(boundary);
            }

            ++t;
        }

        /* Include the L2 norm of feature weights to the objective. */
        logp -= lambda / 2 * crf1mt->norm * scale * scale;

        /* Prevent the scale factor being too small. */
        if (scale < 1e-20) {
            scale_weights(crf1mt->features, K, scale);
            decay = 1.;
            proj = 1.;
        }

        /* One epoch finished. */
        if (!calibration) {
    	    logging(crf1mt->lg, "Log-likelihood: %f\n", logp);
    	    logging(crf1mt->lg, "Feature L2-norm: %f\n", sqrt(crf1mt->norm) * scale);
    	    logging(crf1mt->lg, "Learning rate (eta): %f\n", eta);
            logging(crf1mt->lg, "Total number of feature updates: %f\n", t);
      	    logging(crf1mt->lg, "Seconds required for this iteration: %.3f\n", (clock() - clk_prev) / (double)CLOCKS_PER_SEC);

	        /* Send the tagger with the current parameters. */
	        if (crf1mt->cbe_proc != NULL) {
		        /* Callback notification with the tagger object. */
		        int ret = crf1mt->cbe_proc(crf1mt->cbe_instance, &crf1mt->tagger);
	        }

	        logging(crf1mt->lg, "\n");
        }
    }

    return logp;
}

static floatval_t
l2sgd_calibration(
    crf1ml_t* crf1mt,
    int num_candidates,
    int num_samples,
    floatval_t init_eta,
    floatval_t lambda,
    floatval_t factor
    )
{
    int i;
    int *perm = NULL;
    int dec = 0, ok, trials = 1;
    floatval_t logp;
    floatval_t init_logp = 0.;
    floatval_t best_logp = -DBL_MAX;
    floatval_t eta = init_eta;
    floatval_t best_eta = init_eta;
    const int N = crf1mt->num_sequences;
    const int M = MIN(N, num_samples);

	logging(crf1mt->lg, "Calibrating the learning rate (eta)\n");

	/* Initialize feature weights as zero. */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt->features[i].w = 0;
	}
    crf1mt->norm = 0.;

    /* Initialize a permutation that shuffles the instances. */
    perm = (int*)malloc(sizeof(int) * N);
    crf1ml_shuffle(perm, N, 1);

    /* Compute the initial log likelihood. */
    init_logp = compute_loglikelihood(crf1mt, perm, M, lambda);
	logging(crf1mt->lg, "Initial Log-likelihood: %f\n", init_logp);
    logging(crf1mt->lg, "\n");

    while (num_candidates > 0 || !dec) {
        logging(crf1mt->lg, "***** Trial #%d *****\n", trials);
        logging(crf1mt->lg, "Learning rate (eta): %f\n", eta);

	    /* Initialize feature weights as zero. */
        crf1mt->norm = 0.;
	    for (i = 0;i < crf1mt->num_features;++i) {
		    crf1mt->features[i].w = 0;
	    }

        logp = l2sgd(crf1mt, perm, M, 1.0 / (lambda * eta), lambda, 1, 1);
        ok = (init_logp < logp);

        if (ok) {
    	    logging(crf1mt->lg, "Log-likelihood (good): %f\n", logp);
        } else {
    	    logging(crf1mt->lg, "Log-likelihood (bad): %f\n", logp);
        }
        logging(crf1mt->lg, "\n");

        if (ok) {
            --num_candidates;
            if (best_logp < logp) {
                best_logp = logp;
                best_eta = eta;
            }
        }

        if (!dec) {
            if (ok) {
                eta *= factor;
            } else {
                dec = 1;
                eta = init_eta / factor;
            }
        } else {
            eta /= factor;
        }

        ++trials;
    }

    free(perm);

    best_eta /= factor;

    logging(crf1mt->lg, "***** Calibration result *****\n", trials);
    logging(crf1mt->lg, "Learning rate (eta): %f\n", best_eta);
    logging(crf1mt->lg, "t0: %f\n", 1.0 / (lambda * best_eta));
    logging(crf1mt->lg, "\n");

    return 1.0 / (lambda * best_eta);
}

int crf1ml_lbfgs_sgd(
    crf1ml_t* crf1mt,
    crf1ml_option_t *opt
    )
{
    int i;
    int *perm = NULL;
    floatval_t logp = 0;
    clock_t clk_begin, clk_prev;
	const int N = crf1mt->num_sequences;
    const int K = crf1mt->num_features;

//    const double t0 = 20120;
    // floatval_t t0 = 89360;
    floatval_t t0;
    //const double t0 = 10.120;
    const floatval_t lambda = 1.0 / (1 * N);
    floatval_t eta;

	logging(crf1mt->lg, "Stochastic Gradient Descent (SGD)\n");
    logging(crf1mt->lg, "lambda: %f\n", lambda);
	logging(crf1mt->lg, "\n");
    clk_begin = clock();

	/*
		Initialize feature weights as zero.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt->features[i].w = 0;
	}
    crf1mt->norm = 0.;

    /*
        Initialize a permutation that shuffles the instances.
     */
    perm = (int*)malloc(sizeof(int) * N);
    crf1ml_shuffle(perm, N, 1);

    t0 = l2sgd_calibration(crf1mt, 10, 1000, 0.1, lambda, 2);

    /* Initialize feature weights as zero. */
    crf1mt->norm = 0.;
    for (i = 0;i < crf1mt->num_features;++i) {
	    crf1mt->features[i].w = 0;
    }

    l2sgd(crf1mt, perm, N, t0, lambda, 10, 0);

	logging(crf1mt->lg, "Total seconds required for SGD: %.3f\n", (clock() - clk_begin) / (double)CLOCKS_PER_SEC);
	logging(crf1mt->lg, "\n");

    free(perm);

    return 0;
}

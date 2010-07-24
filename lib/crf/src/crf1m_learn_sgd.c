/*
 *      Training linear-chain CRFs with Stochastic Gradient Descent (SGD).
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

/*
    SGD for L2-regularized MAP estimation.

    The iterative algorithm is inspired by Pegasos:

    Shai Shalev-Shwartz, Yoram Singer, and Nathan Srebro.
    Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
    In Proc. of ICML 2007, pp 807-814, 2007.

    The calibration strategy is inspired by the implementation of sgd:
    http://leon.bottou.org/projects/sgd
    written by Léon Bottou.

    The objective function to minimize is:
        
        f(w) = (lambda/2) * ||w||^2 + (1/N) * \sum_i^N log P^i(y|x)
        lambda = 2 * C / N

    The original version of the Pegasos algorithm.

    0) Initialization
        t = t0
        k = [the batch size]
    1) Computing the learning rate (eta).
        eta = 1 / (lambda * t)
    2) Updating feature weights.
        w = (1 - eta * lambda) w - (eta / k) \sum_i (oexp - mexp)
    3) Projecting feature weights within an L2-ball.
        w = min{1, (1/sqrt(lambda))/||w||} * w
    4) Goto 1 until convergence.

    A naive implementation requires O(K) computations for steps 2 and 3,
    where K is the total number of features. This code implements the procedure
    in an efficient way:

    0) Initialization
        norm2 = 0
        decay = 1
        proj = 1
    1) Computing various factors
        eta = 1 / (lambda * t)
        decay *= (1 - eta * lambda)
        scale = decay * proj
        gain = (eta / k) / scale
    2) Updating feature weights
        Updating feature weights from observation expectation:
            delta = gain * (1.0) * f(x,y)
            norm2 += delta * (delta + w + w);
            w += delta
        Updating feature weights from model expectation:
            delta = gain * (-P(y|x)) * f(x,y)
            norm2 += delta * (delta + w + w);
            w += delta
    3) Projecting feature weights within an L2-ball
        If 1.0 / lambda < norm2 * scale * scale:
            proj = 1.0 / (sqrt(norm2 * lambda) * scale)
    4) Goto 1 until convergence.
*/


#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <crfsuite.h>

#include "logging.h"
#include "params.h"
#include "crf1m.h"

#define MIN(a, b)   ((a) < (b) ? (a) : (b))

typedef struct {
    /** The square of the feature L2 norm. */
    floatval_t  norm2;
    /** Scaling factor for updating weights. */
    floatval_t  gain;
} sgd_internal_t;

#define SGD_INTERNAL(crf1ml)    ((sgd_internal_t*)((crf1ml)->solver_data))

inline void initialize_weights(crf1ml_t* crf1ml)
{
    int i;
    floatval_t *w = crf1ml->w;
    sgd_internal_t *sgdi = SGD_INTERNAL(crf1ml);

    for (i = 0;i < crf1ml->num_features;++i) {
        w[i] = 0.;
    }
    sgdi->norm2 = 0;
}

inline void
update_weight(
    sgd_internal_t* sgdi,
    floatval_t* w,
    const int fid,
    floatval_t a
    )
{
    floatval_t w0 = w[fid];
    w[fid] += a;
    sgdi->norm2 += a * (a + w0 + w0);
}

inline static void update_feature_weights(
    crf1ml_feature_t* f,
    const int fid,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* trainer,
    const crf_sequence_t* seq,
    int t
    )
{
    floatval_t *w = trainer->w;
    sgd_internal_t* sgdi = SGD_INTERNAL(trainer);

    /* Subtract the model expectation from the weight. */
    update_weight(sgdi, w, fid, -sgdi->gain * prob * scale);

    switch (f->type) {
    case FT_STATE:      /**< State features. */
        if (f->dst == seq->items[t].label) {
            update_weight(sgdi, w, fid, sgdi->gain * scale);
        }
        break;
    case FT_TRANS:      /**< Transition features. */
        if (f->src == seq->items[t].label &&
            f->dst == seq->items[t+1].label) {
            update_weight(sgdi, w, fid, sgdi->gain * scale);
        }
        break;
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
    const floatval_t* w = crf1mt->w;
    const int K = crf1mt->num_features;

    /* Set transition scores. */
    crf1mc_reset(crf1mt->ctx, RF_TRANS);
    crf1ml_transition_score(crf1mt, w, K);
    crf1mc_exp_transition(crf1mt->ctx);

    for (i = 0;i < N;++i) {
        seq = &crf1mt->seqs[perm[i]];

        /* Set label sequences and state scores. */
        crf1ml_set_labels(crf1mt, seq);
        crf1mc_reset(crf1mt->ctx, RF_STATE);
        crf1ml_state_score(crf1mt, seq, w, K);
        crf1mc_exp_state(crf1mt->ctx);

        /* Compute forward/backward scores. */
        crf1mc_alpha_score(crf1mt->ctx);
        crf1mc_beta_score(crf1mt->ctx);
        crf1mc_marginal(crf1mt->ctx);

        /* Compute the probability of the input sequence on the model. */
        logp += crf1mc_score(crf1mt->ctx) - crf1mc_lognorm(crf1mt->ctx);
    }

    /* Compute the L2 norm of feature weights. */
    for (i = 0;i < K;++i) {
        norm += w[i] * w[i];
    }

    return logp - 0.5 * lambda * norm * N;
}



static int l2sgd(
    crf1ml_t* crf1mt,
    int *perm,
    const int N,
    const floatval_t t0,
    const floatval_t lambda,
    const int num_epochs,
    int calibration,
    int period,
    const floatval_t epsilon,
    floatval_t *ptr_logp
    )
{
    int i, epoch, ret = 0;
    floatval_t t = 0;
    floatval_t logp = 0;
    floatval_t best_logp = -DBL_MAX;
    clock_t clk_prev;
    crf_sequence_t *seq, *seqs = crf1mt->seqs;
    floatval_t* w = crf1mt->w;
    floatval_t* best_w = NULL;
    const int K = crf1mt->num_features;
    floatval_t eta, scale, boundary;
    floatval_t decay = 1., proj = 1.;
    sgd_internal_t* sgdi = SGD_INTERNAL(crf1mt);
    floatval_t improvement = 0.;
    floatval_t *pf = NULL;

    if (!calibration) {
        pf = (floatval_t*)malloc(sizeof(floatval_t) * period);
        best_w = (floatval_t*)malloc(sizeof(floatval_t) * K);
        for (i = 0;i < K;++i) {
            best_w[i] = 0.;
        }
    }

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
            sgdi->gain = eta / scale;

            /* Set transition scores. */
            crf1mc_reset(crf1mt->ctx, RF_TRANS);
            crf1ml_transition_score_scaled(crf1mt, w, K, scale);
            crf1mc_exp_transition(crf1mt->ctx);

            /* Set label sequences and state scores. */
            crf1ml_set_labels(crf1mt, seq);
            crf1mc_reset(crf1mt->ctx, RF_STATE);
            crf1ml_state_score_scaled(crf1mt, seq, w, K, scale);
            crf1mc_exp_state(crf1mt->ctx);

            /* Compute forward/backward scores. */
            crf1mc_alpha_score(crf1mt->ctx);
            crf1mc_beta_score(crf1mt->ctx);
            crf1mc_marginal(crf1mt->ctx);

            /* Compute the probability of the input sequence on the model. */
            logp += crf1mc_score(crf1mt->ctx) - crf1mc_lognorm(crf1mt->ctx);

            /* Update the feature weights. */
            crf1ml_enum_features(crf1mt, seq, update_feature_weights);

            /* Project feature weights in the L2-ball. */
            boundary = sgdi->norm2 * scale * scale * lambda;
            if (1. < boundary) {
                proj = 1.0 / sqrt(boundary);
            }

            ++t;
        }

        /* Terminate when the log probability is abnormal (NaN, -Inf, +Inf). */
        if (!isfinite(logp)) {
            ret = CRFERR_OVERFLOW;
            break;
        }

        /* Include the L2 norm of feature weights to the objective. */
        /* The factor N is necessary because lambda = 2 * C / N. */
        logp -= 0.5 * lambda * sgdi->norm2 * scale * scale * N;

        /* Prevent the scale factor being too small. */
        if (scale < 1e-20) {
            for (i = 0;i < K;++i) {
                w[i] *= scale;
            }
            decay = 1.;
            proj = 1.;
        }

        /* One epoch finished. */
        if (!calibration) {
            /* Check if the current epoch is the best. */
            if (best_logp < logp) {
                best_logp = logp;
                for (i = 0;i < K;++i) {
                    best_w[i] = w[i];
                }
            }

            /* We don't test the stopping criterion while period < epoch. */
            if (period < epoch) {
                improvement = (pf[(epoch-1) % period] - logp) / logp;
            } else {
                improvement = epsilon;
            }

            /* Store the current value of the objective function. */
            pf[(epoch-1) % period] = logp;

            logging(crf1mt->lg, "Log-likelihood: %f\n", logp);
            if (period < epoch) {
                logging(crf1mt->lg, "Improvement ratio: %f\n", improvement);
            }
            logging(crf1mt->lg, "Feature L2-norm: %f\n", sqrt(sgdi->norm2) * scale);
            logging(crf1mt->lg, "Learning rate (eta): %f\n", eta);
            logging(crf1mt->lg, "Total number of feature updates: %.0f\n", t);
            logging(crf1mt->lg, "Seconds required for this iteration: %.3f\n", (clock() - clk_prev) / (double)CLOCKS_PER_SEC);

            /* Send the tagger with the current parameters. */
            if (crf1mt->cbe_proc != NULL) {
                /* Callback notification with the tagger object. */
                int ret = crf1mt->cbe_proc(crf1mt->cbe_instance, &crf1mt->tagger);
            }

            logging(crf1mt->lg, "\n");

            /* Check for the stopping criterion. */
            if (improvement < epsilon) {
                break;
            }
        }
    }

    /* Restore the best weights. */
    if (best_w != NULL) {
        logp = best_logp;
        for (i = 0;i < K;++i) {
            w[i] = best_w[i];
        }
    }

    free(best_w);
    free(pf);

    if (ptr_logp != NULL) {
        *ptr_logp = logp;
    }
    return ret;
}

static floatval_t
l2sgd_calibration(
    crf1ml_t* crf1mt,
    const crf1ml_sgd_option_t* opt
    )
{
    int *perm = NULL;
    int dec = 0, ok, trials = 1;
    int num_candidates = opt->calibration_candidates;
    clock_t clk_begin = clock();
    floatval_t logp;
    floatval_t init_logp = 0.;
    floatval_t best_logp = -DBL_MAX;
    floatval_t eta = opt->calibration_eta;
    floatval_t best_eta = opt->calibration_eta;
    floatval_t *w = crf1mt->w;
    const int N = crf1mt->num_sequences;
    const int M = MIN(N, opt->calibration_samples);
    const floatval_t init_eta = opt->calibration_eta;
    const floatval_t rate = opt->calibration_rate;
    const floatval_t lambda = opt->lambda;

    logging(crf1mt->lg, "Calibrating the learning rate (eta)\n");
    logging(crf1mt->lg, "sgd.calibration.eta: %f\n", eta);
    logging(crf1mt->lg, "sgd.calibration.rate: %f\n", rate);
    logging(crf1mt->lg, "sgd.calibration.samples: %d\n", M);
    logging(crf1mt->lg, "sgd.calibration.candidates: %d\n", num_candidates);

    /* Initialize a permutation that shuffles the instances. */
    perm = (int*)malloc(sizeof(int) * N);
    crf1ml_shuffle(perm, N, 1);

    /* Initialize feature weights as zero. */
    initialize_weights(crf1mt);

    /* Compute the initial log likelihood. */
    init_logp = compute_loglikelihood(crf1mt, perm, M, lambda);
    logging(crf1mt->lg, "Initial Log-likelihood: %f\n", init_logp);

    while (num_candidates > 0 || !dec) {
        logging(crf1mt->lg, "Trial #%d (eta = %f): ", trials, eta);

        /* Initialize feature weights as zero. */
        initialize_weights(crf1mt);

        /* Perform SGD for one epoch. */
        l2sgd(crf1mt, perm, M, 1.0 / (lambda * eta), lambda, 1, 1, 1, 0., &logp);

        /* Make sure that the learning rate decreases the log-likelihood. */
        ok = isfinite(logp) && (init_logp < logp);
        if (ok) {
            logging(crf1mt->lg, "%f\n", logp);
        } else {
            logging(crf1mt->lg, "%f (worse)\n", logp);
        }

        if (ok) {
            --num_candidates;
            if (best_logp < logp) {
                best_logp = logp;
                best_eta = eta;
            }
        }

        if (!dec) {
            if (ok) {
                eta *= rate;
            } else {
                dec = 1;
                eta = init_eta / rate;
            }
        } else {
            eta /= rate;
        }

        ++trials;
    }

    eta = best_eta;
    logging(crf1mt->lg, "Best learning rate (eta): %f\n", eta);
    logging(crf1mt->lg, "Seconds required: %.3f\n", (clock() - clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crf1mt->lg, "\n");

    free(perm);

    return 1.0 / (lambda * eta);
}

int crf1ml_sgd_options(crf_params_t* params, crf1ml_option_t* opt, int mode)
{
    crf1ml_sgd_option_t* sgd = &opt->sgd;

    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_FLOAT(
            "regularization.sigma", sgd->sigma, 1.,
            ""
            )
        DDX_PARAM_INT(
            "sgd.max_iterations", sgd->max_iterations, 1000,
            ""
            )
        DDX_PARAM_INT(
            "sgd.period", sgd->period, 10,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.delta", sgd->delta, 1e-6,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.calibration.eta", sgd->calibration_eta, 0.1,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.calibration.rate", sgd->calibration_rate, 2.,
            ""
            )
        DDX_PARAM_INT(
            "sgd.calibration.samples", sgd->calibration_samples, 1000,
            ""
            )
        DDX_PARAM_INT(
            "sgd.calibration.candidates", sgd->calibration_candidates, 10,
            ""
            )
    END_PARAM_MAP()

    return 0;
}

int crf1ml_sgd(
    crf1ml_t* crf1mt,
    crf1ml_option_t *opt
    )
{
    int ret = 0;
    int *perm = NULL;
    clock_t clk_begin;
    floatval_t logp = 0;
    const int N = crf1mt->num_sequences;
    const int K = crf1mt->num_features;
    crf1ml_sgd_option_t* sgdopt = &opt->sgd;
    sgd_internal_t sgd_internal;

    /* Set the solver-specific information. */
    crf1mt->solver_data = &sgd_internal;

    sgdopt->lambda = 1.0 / (sgdopt->sigma * sgdopt->sigma * N);

    logging(crf1mt->lg, "Stochastic Gradient Descent (SGD)\n");
    logging(crf1mt->lg, "regularization.sigma: %f\n", sgdopt->sigma);
    logging(crf1mt->lg, "sgd.max_iterations: %d\n", sgdopt->max_iterations);
    logging(crf1mt->lg, "sgd.period: %d\n", sgdopt->period);
    logging(crf1mt->lg, "sgd.delta: %f\n", sgdopt->delta);
    logging(crf1mt->lg, "\n");
    clk_begin = clock();

    /* Calibrate the training rate (eta). */
    sgdopt->t0 = l2sgd_calibration(crf1mt, sgdopt);

    /* Initialize a permutation that shuffles the instances. */
    perm = (int*)malloc(sizeof(int) * N);
    crf1ml_shuffle(perm, N, 1);

    /* Initialize feature weights as zero. */
    initialize_weights(crf1mt);

    /* Perform stochastic gradient descent. */
    ret = l2sgd(
        crf1mt,
        perm,
        N,
        sgdopt->t0,
        sgdopt->lambda,
        sgdopt->max_iterations,
        0,
        sgdopt->period,
        sgdopt->delta,
        &logp
        );

    logging(crf1mt->lg, "Log-likelihood: %f\n", logp);
    logging(crf1mt->lg, "Total seconds required for SGD: %.3f\n", (clock() - clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crf1mt->lg, "\n");

    free(perm);

    return ret;
}

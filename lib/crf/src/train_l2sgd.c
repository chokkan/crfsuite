/*
 *      Online training with L2-regularized Stochastic Gradient Descent (SGD).
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
#include "crfsuite_internal.h"

#include "logging.h"
#include "params.h"
#include "crf1d.h"
#include "vecmath.h"

#define MIN(a, b)   ((a) < (b) ? (a) : (b))

typedef struct {
    floatval_t  sigma;
    floatval_t  lambda;
    floatval_t  t0;
    int         max_iterations;
    int         period;
    floatval_t  delta;
    floatval_t  calibration_eta;
    floatval_t  calibration_rate;
    int         calibration_samples;
    int         calibration_candidates;
} training_option_t;

#if 0
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
#endif


static int l2sgd(
    graphical_model_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    floatval_t *w,
    logging_t *lg,
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
    int i, j, epoch, ret = 0, s = 0;
    floatval_t t = 0;
    floatval_t loss = 0, sum_loss = 0;
    floatval_t best_sum_loss = DBL_MAX;
    clock_t clk_prev;
    const crf_instance_t *seq;
    floatval_t* best_w = NULL;
    const int K = gm->num_features;
    floatval_t eta, scale, gain;
    floatval_t decay = 1., proj = 1.;
    floatval_t improvement = 0.;
    floatval_t norm2 = 0.;
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
            logging(lg, "***** Epoch #%d *****\n", epoch);
        }
        clk_prev = clock();

        if (!calibration) {
            /* Shuffle the instances. */
            dataset_shuffle(trainset);
        }

        /* Loop for instances. */
        sum_loss = 0.;
        for (i = 0;i < N;++i) {
            const crf_instance_t *seq = dataset_get(trainset, i);

            /* Update various factors. */
            eta = 1 / (lambda * (t0 + t));
            decay *= (1.0 - eta * lambda);
            scale = decay * proj;
            gain = eta / scale;

            /*
            if (!calibration) {
                for (j = 0;j < K;++j) {
                    printf("w[%d] = %f\n", j, w[j]);
                }
            }
            */

            gm->set_weights(gm, w);
            gm->objective_and_gradients(gm, seq, &loss, w, scale, gain);

            /*
            if (!calibration) {
                for (j = 0;j < K;++j) {
                    printf("weight[%d] = %f\n", j, w[j]);
                }
            }
            */

            sum_loss += loss;
            ++t;
        }


        /* Terminate when the log probability is abnormal (NaN, -Inf, +Inf). */
        if (!isfinite(loss)) {
            ret = CRFERR_OVERFLOW;
            break;
        }

        vecscale(w, scale, K);
        scale = 1.;
        decay = 1.;

        /* Include the L2 norm of feature weights to the objective. */
        /* The factor N is necessary because lambda = 2 * C / N. */
        norm2 =  vecdot(w, w, K);
        sum_loss += 0.5 * lambda * norm2 * N;

        /* One epoch finished. */
        if (!calibration) {
            /* Check if the current epoch is the best. */
            if (sum_loss < best_sum_loss) {
                best_sum_loss = sum_loss;
                veccopy(best_w, w, K);
            }

            /* We don't test the stopping criterion while period < epoch. */
            if (period < epoch) {
                improvement = (pf[(epoch-1) % period] - sum_loss) / sum_loss;
            } else {
                improvement = epsilon;
            }

            /* Store the current value of the objective function. */
            pf[(epoch-1) % period] = sum_loss;

            logging(lg, "Loss: %f\n", sum_loss);
            if (period < epoch) {
                logging(lg, "Improvement ratio: %f\n", improvement);
            }
            logging(lg, "Feature L2-norm: %f\n", sqrt(norm2));
            logging(lg, "Learning rate (eta): %f\n", eta);
            logging(lg, "Total number of feature updates: %.0f\n", t);
            logging(lg, "Seconds required for this iteration: %.3f\n", (clock() - clk_prev) / (double)CLOCKS_PER_SEC);

            if (testset != NULL) {
                holdout_evaluation(gm, testset, w, lg);
            }
            logging(lg, "\n");

            /* Check for the stopping criterion. */
            if (improvement < epsilon) {
                break;
            }
        }
    }

    /* Restore the best weights. */
    if (best_w != NULL) {
        sum_loss = best_sum_loss;
        veccopy(w, best_w, K);
    }

    free(best_w);
    free(pf);

    if (ptr_logp != NULL) {
        *ptr_logp = sum_loss;
    }
    return ret;
}

static floatval_t
l2sgd_calibration(
    graphical_model_t *gm,
    dataset_t *ds,
    floatval_t *w,
    logging_t *lg,
    const training_option_t* opt
    )
{
    int i, s;
    int *perm = NULL;
    int dec = 0, ok, trials = 1;
    int num = opt->calibration_candidates;
    clock_t clk_begin = clock();
    floatval_t loss;
    floatval_t init_loss = 0.;
    floatval_t best_loss = DBL_MAX;
    floatval_t eta = opt->calibration_eta;
    floatval_t best_eta = opt->calibration_eta;
    const int N = ds->num_instances;
    const int S = MIN(N, opt->calibration_samples);
    const int K = gm->num_features;
    const floatval_t init_eta = opt->calibration_eta;
    const floatval_t rate = opt->calibration_rate;
    const floatval_t lambda = opt->lambda;

    logging(lg, "Calibrating the learning rate (eta)\n");
    logging(lg, "sgd.calibration.eta: %f\n", eta);
    logging(lg, "sgd.calibration.rate: %f\n", rate);
    logging(lg, "sgd.calibration.samples: %d\n", S);
    logging(lg, "sgd.calibration.candidates: %d\n", num);

    /* Initialize a permutation that shuffles the instances. */
    dataset_shuffle(ds);

    /* Initialize feature weights as zero. */
    memset(w, 0, sizeof(w[0]) * K);

    /* Compute the initial log likelihood. */
    gm->set_weights(gm, w);
    s = 0;
    init_loss = 0;
    for (i = 0;i < S;++i) {
        floatval_t loss = 0;
        const crf_instance_t *inst = dataset_get(ds, i);
        gm->objective(gm, inst, &loss);
        init_loss += loss;
    }
    init_loss += 0.5 * lambda * vecdot(w, w, K) * N;

    logging(lg, "Initial loss: %f\n", init_loss);

    while (num > 0 || !dec) {
        logging(lg, "Trial #%d (eta = %f): ", trials, eta);

        /* Initialize feature weights as zero. */
        memset(w, 0, sizeof(w[0]) * K);

        /* Perform SGD for one epoch. */
        l2sgd(
            gm,
            ds,
            NULL,
            w,
            lg,
            S, 1.0 / (lambda * eta), lambda, 1, 1, 1, 0., &loss);

        /* Make sure that the learning rate decreases the log-likelihood. */
        ok = isfinite(loss) && (loss < init_loss);
        if (ok) {
            logging(lg, "%f\n", loss);
        } else {
            logging(lg, "%f (worse)\n", loss);
        }

        if (ok) {
            --num;
            if (loss < best_loss) {
                best_loss = loss;
                best_eta = eta;
            }
        }

        if (!dec) {
            if (ok && 0 < num) {
                eta *= rate;
            } else {
                dec = 1;
                num = opt->calibration_candidates;
                eta = init_eta / rate;
            }
        } else {
            eta /= rate;
        }

        ++trials;
    }

    eta = best_eta;
    logging(lg, "Best learning rate (eta): %f\n", eta);
    logging(lg, "Seconds required: %.3f\n", (clock() - clk_begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    free(perm);

    return 1.0 / (lambda * eta);
}

int exchange_options(crf_params_t* params, training_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_FLOAT(
            "regularization.sigma", opt->sigma, 1.,
            ""
            )
        DDX_PARAM_INT(
            "sgd.max_iterations", opt->max_iterations, 1000,
            ""
            )
        DDX_PARAM_INT(
            "sgd.period", opt->period, 10,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.delta", opt->delta, 1e-6,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.calibration.eta", opt->calibration_eta, 0.1,
            ""
            )
        DDX_PARAM_FLOAT(
            "sgd.calibration.rate", opt->calibration_rate, 2.,
            ""
            )
        DDX_PARAM_INT(
            "sgd.calibration.samples", opt->calibration_samples, 1000,
            ""
            )
        DDX_PARAM_INT(
            "sgd.calibration.candidates", opt->calibration_candidates, 10,
            ""
            )
    END_PARAM_MAP()

    return 0;
}

void crf_train_l2sgd_init(crf_params_t* params)
{
    exchange_options(params, NULL, 0);
}

int crf_train_l2sgd(
    graphical_model_t *gm,
    dataset_t *trainset,
    dataset_t *testset,
    crf_params_t *params,
    logging_t *lg,
    floatval_t **ptr_w
    )
{
    int ret = 0;
    int *perm = NULL;
    floatval_t *w = NULL;
    clock_t clk_begin;
    floatval_t logp = 0;
    const int N = trainset->num_instances;
    const int K = gm->num_features;
    const int T = gm->cap_items;
    training_option_t opt;

    /* Obtain parameter values. */
    exchange_options(params, &opt, -1);

    /* Allocate arrays. */
    w = (floatval_t*)calloc(sizeof(floatval_t), K);

    opt.lambda = 1.0 / (opt.sigma * opt.sigma * N);

    logging(lg, "Stochastic Gradient Descent (SGD)\n");
    logging(lg, "regularization.sigma: %f\n", opt.sigma);
    logging(lg, "sgd.max_iterations: %d\n", opt.max_iterations);
    logging(lg, "sgd.period: %d\n", opt.period);
    logging(lg, "sgd.delta: %f\n", opt.delta);
    logging(lg, "\n");
    clk_begin = clock();

    /* Calibrate the training rate (eta). */
    opt.t0 = l2sgd_calibration(gm, trainset, w, lg, &opt);

    /* Initialize a permutation that shuffles the instances. */
    perm = (int*)malloc(sizeof(int) * N);

    memset(w, 0, sizeof(w[0]) * K);

    /* Perform stochastic gradient descent. */
    ret = l2sgd(
        gm,
        trainset,
        testset,
        w,
        lg,
        N,
        opt.t0,
        opt.lambda,
        opt.max_iterations,
        0,
        opt.period,
        opt.delta,
        &logp
        );

    logging(lg, "Log-likelihood: %f\n", logp);
    logging(lg, "Total seconds required for SGD: %.3f\n", (clock() - clk_begin) / (double)CLOCKS_PER_SEC);
    logging(lg, "\n");

    *ptr_w = w;
    free(perm);

    return ret;
}

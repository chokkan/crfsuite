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

inline static void update_features(
    crf1ml_feature_t* f,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* trainer,
    const crf_sequence_t* seq,
    int t
    )
{
    f->w -= trainer->gain * prob * scale;

    switch (f->type) {
    case FT_STATE:      /**< State features. */
        if (f->dst == seq->items[t].label) {
            f->w += trainer->gain * scale;
        }
        break;
    case FT_TRANS:      /**< Transition features. */
        if (f->src == seq->items[t].label &&
            f->dst == seq->items[t+1].label) {
            f->w += trainer->gain * scale;
        }
        break;
    case FT_TRANS_BOS:  /**< BOS transition features. */
        if (f->dst == seq->items[t].label) {
            f->w += trainer->gain * scale;
        }
        break;
    case FT_TRANS_EOS:  /**< EOS transition features. */
        if (f->src == seq->items[t].label) {
            f->w += trainer->gain * scale;
        }
        break;
    }

    if (trainer->maxw < fabs(f->w)) {
        trainer->maxw = fabs(f->w);
    }

}

inline static void update_features_with_projection(
    crf1ml_feature_t* f,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* trainer,
    const crf_sequence_t* seq,
    int t
    )
{
    trainer->norm -= (f->w * f->w);
    f->w -= trainer->gain * prob * scale;
    trainer->norm += (f->w * f->w);

    switch (f->type) {
    case FT_STATE:      /**< State features. */
        if (f->dst == seq->items[t].label) {
            trainer->norm -= (f->w * f->w);
            f->w += trainer->gain * scale;
            trainer->norm += (f->w * f->w);
        }
        break;
    case FT_TRANS:      /**< Transition features. */
        if (f->src == seq->items[t].label &&
            f->dst == seq->items[t+1].label) {
            trainer->norm -= (f->w * f->w);
            f->w += trainer->gain * scale;
            trainer->norm += (f->w * f->w);
        }
        break;
    case FT_TRANS_BOS:  /**< BOS transition features. */
        if (f->dst == seq->items[t].label) {
            trainer->norm -= (f->w * f->w);
            f->w += trainer->gain * scale;
            trainer->norm += (f->w * f->w);
        }
        break;
    case FT_TRANS_EOS:  /**< EOS transition features. */
        if (f->src == seq->items[t].label) {
            trainer->norm -= (f->w * f->w);
            f->w += trainer->gain * scale;
            trainer->norm += (f->w * f->w);
        }
        break;
    }

    if (trainer->maxw < fabs(f->w)) {
        trainer->maxw = fabs(f->w);
    }
}

floatval_t l2norm(const crf1ml_feature_t* fs, const int n)
{
    int i;
    floatval_t s = 0.;

    for (i = 0;i < n;++i) {
        s += fs[i].w * fs[i].w;
    }
    return sqrt(s);
}

void scale_weights(crf1ml_feature_t* fs, const int n, const floatval_t scale)
{
    int i;
    for (i = 0;i < n;++i) {
        fs[i].w *= scale;
    }
}

int crf1ml_lbfgs_sgd(
    crf1ml_t* crf1mt,
    crf1ml_option_t *opt
    )
{
    int epoch, i, j, k;
    clock_t duration;
    floatval_t logp = 0, loss = 0.;
    floatval_t maxv, v;
	crf_sequence_t *seq, *seqs = crf1mt->seqs;
	const int N = crf1mt->num_sequences;
    const int K = crf1mt->num_features;
    int *perm = NULL;

//    const double t0 = 20120;
    const double t0 = 89360;
    //const double t0 = 10.120;
    double t = 0;
    const double lambda = 1.0 / (1 * N);
    //const double lambda = 0.01;
    double norm, scale;

	logging(crf1mt->lg, "Stochastic Gradient Descent (SGD)\n");
    logging(crf1mt->lg, "Lambda: %f\n", lambda);
    logging(crf1mt->lg, "t0: %f\n", t0);
	logging(crf1mt->lg, "\n");

	/*
		Initialize feature weights as zero.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1ml_feature_t* f = &crf1mt->features[i];
		f->w = 0;
        f->oexp = 0;
        f->mexp = 0;
	}

    /*
        Initialize a permutation that shuffles the instances.
     */
    perm = (int*)malloc(sizeof(int) * N);
    crf1ml_shuffle(perm, N, 1);

    crf1mt->decay = 1.;
    crf1mt->norm = 0.;
    crf1mt->maxw = 1.;

    scale = 1.;

    for (epoch = 1;epoch <= 10;++epoch) {
        crf1mt->clk_prev = clock();

	    logging(crf1mt->lg, "***** Epoch #%d *****\n", epoch);

        /* Generate a permutation that shuffles the instances. */
        crf1ml_shuffle(perm, N, 0);
        loss = 0.;

        for (i = 0;i < N;++i) {
            seq = &seqs[perm[i]];

            crf1mt->eta = 1 / (lambda * (t0 + t));
            crf1mt->decay *= (1.0 - crf1mt->eta * lambda);
            crf1mt->gain = crf1mt->eta / (crf1mt->decay * scale);

            /* Set transition scores. */
            crf1ml_transition_score(crf1mt, 0.1/* / crf1mt->maxw*/);
            crf1mc_exp_transition(crf1mt->ctx);

		    /* Set label sequences and state scores. */
		    crf1ml_set_labels(crf1mt, seq);
		    crf1ml_state_score(crf1mt, seq, 0.1/* / crf1mt->maxw*/);
		    crf1mc_exp_state(crf1mt->ctx);

		    /* Compute forward/backward scores. */
		    crf1mc_forward_score(crf1mt->ctx);
		    crf1mc_backward_score(crf1mt->ctx);

		    /* Compute the probability of the input sequence on the model. */
            logp = crf1mc_logprob(crf1mt->ctx);
            loss += logp;

		    /* Update the model expectations of features. */
		    crf1ml_enum_features(crf1mt, seq, update_features_with_projection);

            /*
            for (j = 0;j < crf1mt->num_features;++j) {
	            crf1ml_feature_t* f = &crf1mt->features[j];
                f->w *= (1.0 - eta * lambda);
                f->w += eta * (f->oexp - f->mexp);
                f->oexp = 0;
                f->mexp = 0;
            }
            */

            ++t;

            /* Project to the L2-ball. */
            norm = crf1mt->norm * crf1mt->decay * crf1mt->decay * scale * scale;
            if (norm * lambda > 1.) {
                scale = sqrt(1.0/(lambda*norm));
            }

            /*
            norm = l2norm(crf1mt->features, crf1mt->num_features);
            scale = MIN(1, 1 / (sqrt(lambda) * norm));
            scale_weights(crf1mt->features, crf1mt->num_features, scale);
            */
        }

        /* Prevent the decay factor being too small. */
        printf("decay = %f\n", crf1mt->decay);
        if (crf1mt->decay < 1e-6) {
            logging(crf1mt->lg, "Merging the decay factor to the weights\n");
            scale_weights(crf1mt->features, K, crf1mt->decay);
            crf1mt->decay = 1.;
        }

	    duration = clock() - crf1mt->clk_prev;

        norm = 0.;
        for (k = 0;k < K;++k) {
            norm += crf1mt->features[k].w * crf1mt->features[k].w;
        }
        norm = sqrt(norm) * crf1mt->decay * scale;

    	logging(crf1mt->lg, "Log-likelihood: %f\n", loss);
    	logging(crf1mt->lg, "Feature norm: %f\n", norm);
    	logging(crf1mt->lg, "Eta: %f\n", crf1mt->eta);
      	logging(crf1mt->lg, "Seconds required for this iteration: %.3f\n", duration / (double)CLOCKS_PER_SEC);

	    /* Send the tagger with the current parameters. */
	    if (crf1mt->cbe_proc != NULL) {
		    /* Callback notification with the tagger object. */
		    int ret = crf1mt->cbe_proc(crf1mt->cbe_instance, &crf1mt->tagger);
	    }
	    logging(crf1mt->lg, "\n");
    }

	logging(crf1mt->lg, "Total seconds required for SGD: %.3f\n", (clock() - crf1mt->clk_begin) / (double)CLOCKS_PER_SEC);
	logging(crf1mt->lg, "\n");

    free(perm);

    return 0;
}

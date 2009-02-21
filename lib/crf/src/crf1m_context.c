/*
 *      Forward backward algorithm of linear-chain CRF.
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <crf.h>

#include "crf1m.h"

crf1m_context_t* crf1mc_new(int L, int T)
{
	int ret = 0;
	crf1m_context_t* ctx = NULL;
	
	ctx = (crf1m_context_t*)calloc(1, sizeof(crf1m_context_t));
	if (ctx != NULL) {
		ctx->num_labels = L;
		ctx->log_norm = 0;
		ctx->trans_score = (floatval_t*)calloc((L+1) * (L+1), sizeof(floatval_t));
		if (ctx->trans_score == NULL) goto error_exit;

		if (ret = crf1mc_set_num_items(ctx, T)) {
			goto error_exit;
		}
		ctx->num_items = 0;
	}
	return ctx;

error_exit:
	crf1mc_delete(ctx);
	return NULL;
}

int crf1mc_set_num_items(crf1m_context_t* ctx, int T)
{
	const int L = ctx->num_labels;

	ctx->num_items = T;

	if (ctx->max_items < T) {
		free(ctx->backward_edge);
		free(ctx->state_score);
		free(ctx->scale_factor);
		free(ctx->backward_score);
		free(ctx->forward_score);
		free(ctx->labels);

		ctx->labels = (int*)calloc(T, sizeof(int));
		if (ctx->labels == NULL) return CRFERR_OUTOFMEMORY;
		ctx->forward_score = (floatval_t*)calloc((T+1) * L, sizeof(floatval_t));
		if (ctx->forward_score == NULL) return CRFERR_OUTOFMEMORY;
		ctx->scale_factor = (floatval_t*)calloc((T+1), sizeof(floatval_t));
		if (ctx->scale_factor == NULL) return CRFERR_OUTOFMEMORY;
		ctx->backward_score = (floatval_t*)calloc((T+1) * L, sizeof(floatval_t));
		if (ctx->backward_score == NULL) return CRFERR_OUTOFMEMORY;
		ctx->state_score = (floatval_t*)calloc(T * L, sizeof(floatval_t));
		if (ctx->state_score == NULL) return CRFERR_OUTOFMEMORY;
		ctx->backward_edge = (int*)calloc(T * L, sizeof(int));
		if (ctx->backward_edge == NULL) return CRFERR_OUTOFMEMORY;

		ctx->max_items = T;
	}

	return 0;
}

void crf1mc_delete(crf1m_context_t* ctx)
{
	if (ctx != NULL) {
		free(ctx->backward_edge);
		free(ctx->state_score);
		free(ctx->scale_factor);
		free(ctx->backward_score);
		free(ctx->forward_score);
		free(ctx->labels);
		free(ctx->trans_score);
	}
	free(ctx);
}

void crf1mc_exp_state(crf1m_context_t* ctx)
{
	int i, t;
	floatval_t *state = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	for (t = 0;t < T;++t) {
		state = STATE_SCORE_AT(ctx, t);
		for (i = 0;i < L;++i) {
			state[i] = (state[i] == 0. ? 1. : exp(state[i]));
		}
	}
}

void crf1mc_exp_transition(crf1m_context_t* ctx)
{
	int i, j;
	floatval_t *trans = NULL;
	const int L = ctx->num_labels;

	for (i = 0;i <= L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		for (j = 0;j <= L;++j) {
			trans[j] = (trans[j] == 0. ? 1. : exp(trans[j]));
		}
	}
}

void crf1mc_forward_score(crf1m_context_t* ctx)
{
	int i, j, t;
	floatval_t score, sum, *cur = NULL;
	const floatval_t *prev = NULL, *trans = NULL, *state = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/* Compute the score to stay on labels at position #0. */
	sum = 0.;
	cur = FORWARD_SCORE_AT(ctx, 0);
	state = STATE_SCORE_AT(ctx, 0);
	trans = TRANS_SCORE_FROM(ctx, L);
	for (j = 0;j < L;++j) {
		/* Transit from BOS to #j. */
		sum += cur[j] = trans[j] * state[j];
	}
    ctx->scale_factor[0] = (sum != 0.) ? 1. / sum : 1.;
	for (j = 0;j < L;++j) cur[j] *= ctx->scale_factor[0];

	/* Compute the scores at position #t. */
	for (t = 1;t < T;++t) {
		sum = 0.;
		prev = FORWARD_SCORE_AT(ctx, t-1);
		cur = FORWARD_SCORE_AT(ctx, t);
		state = STATE_SCORE_AT(ctx, t);

		/* Compute the score to stay on label #j at position #t. */
		for (j = 0;j < L;++j) {
			score = 0;
			for (i = 0;i < L;++i) {
				/* Transit from #i at #(t-1) to #j at #t. */
				trans = TRANS_SCORE_FROM(ctx, i);
				score += prev[i] * trans[j];
			}
			/* Add the state score on label #j at #t. */
			sum += cur[j] = score * state[j];
		}

		/* Compute the scale factor. */
        ctx->scale_factor[t] = (sum != 0.) ? 1. / sum : 1.;
		/* Apply the scaling factor. */
		for (j = 0;j < L;++j) cur[j] *= ctx->scale_factor[t];
	}

	/* Compute the scores to reach EOS. */
	sum = 0.;
	cur = FORWARD_SCORE_AT(ctx, T-1);
	for (i = 0;i < L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		sum += cur[i] * trans[L];
	}
    ctx->scale_factor[T] = (sum != 0.) ? 1. / sum : 1.;

	/* Compute the logarithm of the normalization factor here.
		norm = sum / (C[0] * C[1] ... * C[T-1]) = 1 / (C[0] * ... * C[T])
		log(norm) = - \sum_{t = 0}^{T} log(C[t]).
	 */
	ctx->log_norm = 0.;
	for (t = 0;t <= T;++t) {
		ctx->log_norm -= log(ctx->scale_factor[t]);
	}
}

void crf1mc_backward_score(crf1m_context_t* ctx)
{
	int i, j, t;
	floatval_t score, scale, *cur = NULL;
	const floatval_t *next = NULL, *state = NULL, *trans = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/* Compute the score to reach EOS from the label #i at position #T-1. */
	cur = BACKWARD_SCORE_AT(ctx, T-1);
	scale = ctx->scale_factor[T-1];
	for (i = 0;i < L;++i) {
		/* Transit from label #i at position #(T-1) to EOS. */
		trans = TRANS_SCORE_FROM(ctx, i);
		cur[i] = trans[L] * scale;
	}

	/* Compute the scores from position #t. */
	for (t = T-2;0 <= t;--t) {
		cur = BACKWARD_SCORE_AT(ctx, t);
		next = BACKWARD_SCORE_AT(ctx, t+1);
		state = STATE_SCORE_AT(ctx, t+1);
		scale = ctx->scale_factor[t];

		/* Compute the score to reach EOS from label #i at position #t. */
		for (i = 0;i < L;++i) {
			score = 0.;
			trans = TRANS_SCORE_FROM(ctx, i);
			for (j = 0;j < L;++j) {
				/* Transit from labels #i to #j at position #(t+1). */
				score += trans[j] * state[j] * next[j];
			}
			cur[i] = score * scale;
		}
	}
}

floatval_t crf1mc_logprob(crf1m_context_t* ctx)
{
	int i, j, t;
	floatval_t ret = 0;
	const floatval_t *state = NULL, *cur = NULL, *trans = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;
	const int *labels = ctx->labels;

	/* Transit from BOS to (0, labels[0]). */
	i = labels[0];
	cur = FORWARD_SCORE_AT(ctx, 0);
	ret = log(cur[i]) - log(ctx->scale_factor[0]);

	/* Loop over the rest of items. */
	for (t = 1;t < T;++t) {
		j = labels[t];
		trans = TRANS_SCORE_FROM(ctx, i);
		state = STATE_SCORE_AT(ctx, t);

		/* Transit from (t-1, i) to (t, j). */
		ret += log(trans[j]);
		ret += log(state[j]);
		i = j;
	}

	/* Transit from (T-1, i) to EOS. */
	cur = BACKWARD_SCORE_AT(ctx, T-1);
	ret += log(cur[i]) - log(ctx->scale_factor[T-1]);

	/* Subtract the logarithm of the normalization factor. */
	ret -= ctx->log_norm;
	return ret;
}

floatval_t crf1mc_viterbi(crf1m_context_t* ctx)
{
	int i, j, t;
	int *back = NULL;
	floatval_t max_score, score, *cur = NULL;
	const floatval_t *prev = NULL, *state = NULL, *trans = NULL;
	int *labels = ctx->labels;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/*
		This function assumes state and trans scores to be in the logarithm domain.
	 */

	/* Compute the score to stay on labels at position #0. */
	cur = FORWARD_SCORE_AT(ctx, 0);
	state = STATE_SCORE_AT(ctx, 0);
	trans = TRANS_SCORE_FROM(ctx, L);
	for (j = 0;j < L;++j) {
		/* Transit from BOS to #j. */
		/* exp(cur[j]) = exp(trans[j]) * exp(state[j]) */
		cur[j] = trans[j] + state[j];
	}

	/* Compute the scores at position #t. */
	for (t = 1;t < T;++t) {
		prev = FORWARD_SCORE_AT(ctx, t-1);
		cur = FORWARD_SCORE_AT(ctx, t);
		state = STATE_SCORE_AT(ctx, t);
		back = BACKWARD_EDGE_AT(ctx, t);

		/* Compute the score to stay on label #j at position #t. */
		for (j = 0;j < L;++j) {
			max_score = -FLOAT_MAX;

			for (i = 0;i < L;++i) {
				/* Transit from #i at #(t-1) to #j at #t. */
				trans = TRANS_SCORE_FROM(ctx, i);
				score = prev[i] + trans[j];

				/* Store this path if it has the maximum score. */
				if (max_score < score) {
					max_score = score;
					/* Backward link (#t, #j) -> (#t-1, #i). */
					back[j] = i;
				}
			}
			/* Add the state score on label #j at #t. */
			cur[j] = max_score + state[j];
		}
	}

	/* Find the node (#T, #i) that reaches EOS with the maximum score. */
	max_score = -FLOAT_MAX;
	prev = FORWARD_SCORE_AT(ctx, T-1);
	for (i = 0;i < L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		score = prev[i] + trans[L];
		if (max_score < score) {
			max_score = score;
			labels[T-1] = i;		/* Tag the item #T. */
		}
	}

	/* Tag labels by tracing the backward links. */
	for (t = T-2;0 <= t;--t) {
		back = BACKWARD_EDGE_AT(ctx, t+1);
		labels[t] = back[labels[t+1]];
	}

	/* Return the maximum score (without the normalization factor subtracted). */
	return max_score;
}

void crf1mc_debug_context(crf1m_context_t* ctx, FILE *fp)
{
	int i, j, t;
	floatval_t scale;
	const floatval_t *fwd = NULL, *bwd = NULL;
	const floatval_t *state = NULL, *trans = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/* Output state score. */
	fprintf(fp, "# ===== State matrix =====\n");
	for (t = 0;t < T;++t) {
		state = STATE_SCORE_AT(ctx, t);

		/* Print the item position. */
		if (t == T) {
			fprintf(fp, "BOS/EOS");
		} else {
			fprintf(fp, "%d", t);
		}

		/* Print the forward/backward scores at the current position. */
		for (i = 0;i < L;++i) {
			printf("\t%1.3e", state[i]);
		}

		printf("\n");
	}
	fprintf(fp, "\n");

	/* Output transition score. */
	fprintf(fp, "# ===== Transition matrix =====\n");
	for (i = 0;i <= L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);

		/* Print the item position. */
		if (i == L) {
			fprintf(fp, "BOS");
		} else {
			fprintf(fp, "%d", i);
		}

		/* Print the forward/backward scores at the current position. */
		for (j = 0;j <= L;++j) {
			printf("\t%1.3e", trans[j]);
		}

		printf("\n");
	}
	fprintf(fp, "\n");

	/* Output forward score. */
	scale = 1.;
	fprintf(fp, "# ===== Forward matrix =====\n");
	for (t = 0;t < T;++t) {
		fwd = FORWARD_SCORE_AT(ctx, t);
		scale *= ctx->scale_factor[t];

		/* Print the item position. */
		fprintf(fp, "%d", t);

		/* Print the forward/backward scores at the current position. */
		for (i = 0;i < L;++i) {
			printf("\t%1.3e", fwd[i] / scale);
		}

		printf("\n");
	}
	fprintf(fp, "\n");

	/* Output forward score. */
	scale = 1.;
	fprintf(fp, "# ===== Backward matrix =====\n");
	for (t = T-1;0 <= t;--t) {
		bwd = BACKWARD_SCORE_AT(ctx, t);
		scale *= ctx->scale_factor[t];

		/* Print the item position. */
		fprintf(fp, "%d", t);

		/* Print the forward/backward scores at the current position. */
		for (i = 0;i < L;++i) {
			printf("\t%1.3e", bwd[i] / scale);
		}

		printf("\n");
	}
	fprintf(fp, "\n");

	fprintf(fp, "# ===== Information =====\n");
	fprintf(fp, "NORM\t%f\n", exp(ctx->log_norm));
}

void crf1mc_test_context(FILE *fp)
{
	crf1m_context_t *ctx = crf1mc_new(3, 3);
	floatval_t *trans = NULL, *state = NULL;
	
	state = STATE_SCORE_AT(ctx, 0);
	state[0] = .4;	state[1] = .5;	state[2] = .1;
	state = STATE_SCORE_AT(ctx, 1);
	state[0] = .4;	state[1] = .1;	state[2] = .5;
	state = STATE_SCORE_AT(ctx, 2);
	state[0] = .4;	state[1] = .1;	state[2] = .5;

	trans = TRANS_SCORE_FROM(ctx, 0);
	trans[0] = .3;	trans[1] = .1;	trans[2] = .4;	trans[3] = .2;
	trans = TRANS_SCORE_FROM(ctx, 1);
	trans[0] = .6;	trans[1] = .2;	trans[2] = .1;	trans[3] = .1;
	trans = TRANS_SCORE_FROM(ctx, 2);
	trans[0] = .5;	trans[1] = .2;	trans[2] = .1;	trans[3] = .2;
	trans = TRANS_SCORE_FROM(ctx, 3);
	trans[0] = .5;	trans[1] = .3;	trans[2] = .2;	trans[3] = .0;

	ctx->num_items = ctx->max_items;
	crf1mc_forward_score(ctx);
	crf1mc_backward_score(ctx);
	crf1mc_debug_context(ctx, fp);

	ctx->labels[0] = 0;	ctx->labels[1] = 2;	ctx->labels[2] = 0;
	printf("PROB\t%f\n", crf1mc_logprob(ctx));
}

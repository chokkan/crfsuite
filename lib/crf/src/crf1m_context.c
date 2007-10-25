/*
 *      Forward backward algorithm of linear-chain CRF.
 *
 * Copyright (c) 2007, Naoaki Okazaki
 *
 * This file is part of libCRF.
 *
 * libCRF is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * any later version.
 *
 * libCRF is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
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

inline static floatval_t logsumexp(floatval_t x, floatval_t y, int flag)
{
	double vmin, vmax;

	if (flag) return y;
	if (x == y) return x + 0.69314718055;	/* log(2) */
	if (x < y) {
		vmin = x; vmax = y;
	} else {
		vmin = y; vmax = x;
	}
	return (vmin + 50 < vmax) ?
		vmax : vmax + log(exp(vmin - vmax) + 1.0);
}

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
		free(ctx->backward_score);
		free(ctx->forward_score);
		free(ctx->labels);

		ctx->labels = (int*)calloc(T, sizeof(int));
		if (ctx->labels == NULL) return CRFERR_OUTOFMEMORY;
		ctx->forward_score = (floatval_t*)calloc((T+1) * L, sizeof(floatval_t));
		if (ctx->forward_score == NULL) return CRFERR_OUTOFMEMORY;
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
		free(ctx->backward_score);
		free(ctx->forward_score);
		free(ctx->labels);
		free(ctx->trans_score);
	}
	free(ctx);
}

void crf1mc_forward_score(crf1m_context_t* ctx)
{
	int i, j, t;
	floatval_t score, *cur = NULL;
	const floatval_t *prev = NULL, *trans = NULL, *state = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/* Initialize the scores to stay on BOS as zero as these values
	   are not updated in this function. */
	cur = FORWARD_SCORE_AT(ctx, T);
	for (i = 0;i < L;++i) {
		cur[i] = 0;
	}

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

		/* Compute the score to stay on label #j at position #t. */
		for (j = 0;j < L;++j) {
			score = 0;
			for (i = 0;i < L;++i) {
				/* Transit from #i at #(t-1) to #j at #t. */
				trans = TRANS_SCORE_FROM(ctx, i);
				score = logsumexp(score, prev[i] + trans[j], i == 0);
			}
			/* Add the state score on label #j at #t. */
			cur[j] = score + state[j];
		}
	}

	/* Compute the logarithm of the normalization factor here. */
	score = 0;
	cur = FORWARD_SCORE_AT(ctx, T-1);
	for (i = 0;i < L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		score = logsumexp(score, cur[i] + trans[L], i == 0);
	}
	ctx->log_norm = score;
}

void crf1mc_backward_score(crf1m_context_t* ctx)
{
	int i, j, t;
	floatval_t score, *cur = NULL;
	const floatval_t *next = NULL, *state = NULL, *trans = NULL;
	const int T = ctx->num_items;
	const int L = ctx->num_labels;

	/* Initialize the scores to stay on BOS as zero as these values
	   are not updated in this function. */
	cur = BACKWARD_SCORE_AT(ctx, T);
	for (i = 0;i < L;++i) {
		cur[i] = 0;
	}

	/* Compute the score to reach EOS from the label #i at position #T-1. */
	cur = BACKWARD_SCORE_AT(ctx, T-1);
	for (i = 0;i < L;++i) {
		/* Transit from label #i at position #(T-1) to EOS. */
		/* exp(cur[i]) = exp(trans[L]) */
		trans = TRANS_SCORE_FROM(ctx, i);
		cur[i] = trans[L];
	}

	/* Compute the scores from position #t. */
	for (t = T-2;0 <= t;--t) {
		cur = BACKWARD_SCORE_AT(ctx, t);
		next = BACKWARD_SCORE_AT(ctx, t+1);
		state = STATE_SCORE_AT(ctx, t+1);

		/* Compute the score to reach EOS from label #i at position #t. */
		for (i = 0;i < L;++i) {
			score = 0;
			trans = TRANS_SCORE_FROM(ctx, i);
			for (j = 0;j < L;++j) {
				/* Transit from labels #i to #j at position #(t+1). */
				/* exp(score) += exp(trans[j]) * exp(state[j]) * exp(next[j]) */
				score = logsumexp(score, trans[j] + state[j] + next[j], j == 0);
			}
			cur[i] = score;
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
	ret = cur[i];

	/* Loop over the rest of items. */
	for (t = 1;t < T;++t) {
		j = labels[t];
		trans = TRANS_SCORE_FROM(ctx, i);
		state = STATE_SCORE_AT(ctx, t);

		/* Transit from (t-1, i) to (t, j). */
		ret += trans[j];
		ret += state[j];
		i = j;
	}

	/* Transit from (T-1, i) to EOS. */
	cur = BACKWARD_SCORE_AT(ctx, T-1);
	ret += cur[i];

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
			printf("\t%1.3e", exp(state[i]));
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
			printf("\t%1.3e", exp(trans[j]));
		}

		printf("\n");
	}
	fprintf(fp, "\n");

	/* Output forward score. */
	fprintf(fp, "# ===== Forward/Backward matrix =====\n");
	for (t = 0;t <= T;++t) {
		fwd = FORWARD_SCORE_AT(ctx, t);
		bwd = BACKWARD_SCORE_AT(ctx, t);

		/* Print the item position. */
		if (t == T) {
			fprintf(fp, "BOS/EOS");
		} else {
			fprintf(fp, "%d", t);
		}

		/* Print the forward/backward scores at the current position. */
		for (i = 0;i < L;++i) {
			printf("\t%1.3e/%1.3e", exp(fwd[i]), exp(bwd[i]));
		}

		printf("\n");
	}
	fprintf(fp, "\n");

}

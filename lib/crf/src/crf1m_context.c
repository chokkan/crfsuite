/*
 *      Forward backward algorithm of linear-chain CRF.
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <crfsuite.h>

#include "crf1m.h"

crf1m_context_t* crf1mc_new(int L, int T)
{
    int ret = 0;
    crf1m_context_t* ctx = NULL;
    
    ctx = (crf1m_context_t*)calloc(1, sizeof(crf1m_context_t));
    if (ctx != NULL) {
        ctx->num_labels = L;
        ctx->log_norm = 0;
        ctx->trans_score = (floatval_t*)calloc(L * L, sizeof(floatval_t));
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

    for (i = 0;i < L;++i) {
        trans = TRANS_SCORE_FROM(ctx, i);
        for (j = 0;j < L;++j) {
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

    /* Compute the alpha scores on nodes (0, *). */
    cur = FORWARD_SCORE_AT(ctx, 0);
    state = STATE_SCORE_AT(ctx, 0);
    for (sum = 0., j = 0;j < L;++j) {
        /* The alpha score of the node (0, j). */
        sum += cur[j] = state[j];
    }
    ctx->scale_factor[0] = (sum != 0.) ? 1. / sum : 1.;
    for (j = 0;j < L;++j) cur[j] *= ctx->scale_factor[0];

    /* Compute the alpha scores on nodes (t, *). */
    for (t = 1;t < T;++t) {
        prev = FORWARD_SCORE_AT(ctx, t-1);
        cur = FORWARD_SCORE_AT(ctx, t);
        state = STATE_SCORE_AT(ctx, t);

        /* Compute the alpha score of the node (t, j). */
        for (sum = 0., j = 0;j < L;++j) {
            for (score = 0., i = 0;i < L;++i) {
                /* Transit from (t-1, i) to (t, j). */
                trans = TRANS_SCORE_FROM(ctx, i);
                score += prev[i] * trans[j];
            }
            /* Add the state score on (t, j). */
            sum += cur[j] = score * state[j];
        }

        /* Compute the scale factor. */
        ctx->scale_factor[t] = (sum != 0.) ? 1. / sum : 1.;
        /* Apply the scaling factor. */
        for (j = 0;j < L;++j) cur[j] *= ctx->scale_factor[t];
    }

    /* Compute the logarithm of the normalization factor here.
        norm = 1. / (C[0] * C[1] ... * C[T-1])
        log(norm) = - \sum_{t = 0}^{T-1} log(C[t]).
     */
    ctx->log_norm = 0.;
    for (t = 0;t < T;++t) {
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

    /* Compute the beta scores at (T-1, *). */
    cur = BACKWARD_SCORE_AT(ctx, T-1);
    scale = ctx->scale_factor[T-1];
    for (i = 0;i < L;++i) {
        /* The beta scores at (T-1, i). */
        cur[i] = scale;
    }

    /* Compute the beta scores at (t, *). */
    for (t = T-2;0 <= t;--t) {
        cur = BACKWARD_SCORE_AT(ctx, t);
        next = BACKWARD_SCORE_AT(ctx, t+1);
        state = STATE_SCORE_AT(ctx, t+1);
        scale = ctx->scale_factor[t];

        /* Compute the beta score at (t, i). */
        for (i = 0;i < L;++i) {
            score = 0.;
            trans = TRANS_SCORE_FROM(ctx, i);
            for (j = 0;j < L;++j) {
                /* Transit from (t, i) to (t+1, j). */
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

    /* Stay at (0, labels[0]). */
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

    /* Compute the scores at (0, *). */
    cur = FORWARD_SCORE_AT(ctx, 0);
    state = STATE_SCORE_AT(ctx, 0);
    for (j = 0;j < L;++j) {
        cur[j] = state[j];
    }

    /* Compute the scores at (t, *). */
    for (t = 1;t < T;++t) {
        prev = FORWARD_SCORE_AT(ctx, t-1);
        cur = FORWARD_SCORE_AT(ctx, t);
        state = STATE_SCORE_AT(ctx, t);
        back = BACKWARD_EDGE_AT(ctx, t);

        /* Compute the score of (t, j). */
        for (j = 0;j < L;++j) {
            max_score = -FLOAT_MAX;

            for (i = 0;i < L;++i) {
                /* Transit from (t-1, i) to (t, j). */
                trans = TRANS_SCORE_FROM(ctx, i);
                score = prev[i] + trans[j];

                /* Store this path if it has the maximum score. */
                if (max_score < score) {
                    max_score = score;
                    /* Backward link (#t, #j) -> (#t-1, #i). */
                    back[j] = i;
                }
            }
            /* Add the state score on (t, j). */
            cur[j] = max_score + state[j];
        }
    }

    /* Find the node (#T, #i) that reaches EOS with the maximum score. */
    max_score = -FLOAT_MAX;
    prev = FORWARD_SCORE_AT(ctx, T-1);
    for (i = 0;i < L;++i) {
        if (max_score < prev[i]) {
            max_score = prev[i];
            labels[T-1] = i;        /* Tag the item #T. */
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

void crf1mc_test_context(FILE *fp)
{
    int y1, y2, y3;
    floatval_t norm = 0;
    const int L = 3;
    const int T = 3;
    const floatval_t eps = 1e-9;
    crf1m_context_t *ctx = crf1mc_new(L, T);
    floatval_t *trans = NULL, *state = NULL;
    floatval_t scores[3][3][3];
    
    /* Initialize the state scores. */
    state = STATE_SCORE_AT(ctx, 0);
    state[0] = .4;    state[1] = .5;    state[2] = .1;
    state = STATE_SCORE_AT(ctx, 1);
    state[0] = .4;    state[1] = .1;    state[2] = .5;
    state = STATE_SCORE_AT(ctx, 2);
    state[0] = .4;    state[1] = .1;    state[2] = .5;

    /* Initialize the transition scores. */
    trans = TRANS_SCORE_FROM(ctx, 0);
    trans[0] = .3;    trans[1] = .1;    trans[2] = .4;
    trans = TRANS_SCORE_FROM(ctx, 1);
    trans[0] = .6;    trans[1] = .2;    trans[2] = .1;
    trans = TRANS_SCORE_FROM(ctx, 2);
    trans[0] = .5;    trans[1] = .2;    trans[2] = .1;

    ctx->num_items = ctx->max_items;
    crf1mc_forward_score(ctx);
    crf1mc_backward_score(ctx);
    /*crf1mc_debug_context(ctx, fp);*/

    /* Compute the score of every label sequence. */
    for (y1 = 0;y1 < L;++y1) {
        floatval_t s1 = STATE_SCORE_AT(ctx, 0)[y1];
        for (y2 = 0;y2 < L;++y2) {
            floatval_t s2 = s1;
            s2 *= TRANS_SCORE_FROM(ctx, y1)[y2];
            s2 *= STATE_SCORE_AT(ctx, 1)[y2];
            for (y3 = 0;y3 < L;++y3) {
                floatval_t s3 = s2;
                s3 *= TRANS_SCORE_FROM(ctx, y2)[y3];
                s3 *= STATE_SCORE_AT(ctx, 2)[y3];
                scores[y1][y2][y3] = s3;
            }
        }
    }

    /* Compute the partition factor. */
    norm = 0.;
    for (y1 = 0;y1 < L;++y1) {
        for (y2 = 0;y2 < L;++y2) {
            for (y3 = 0;y3 < L;++y3) {
                norm += scores[y1][y2][y3];
            }
        }
    }

    /* Check the partition factor. */
    fprintf(fp, "Check for the partition factor... ");
    if (fabs(norm - exp(ctx->log_norm)) < eps) {
        fprintf(fp, "OK (%f)\n", exp(ctx->log_norm));
    } else {
        fprintf(fp, "FAIL: %f (%f)\n", exp(ctx->log_norm), norm);
    }

    /* Compute the sequence probabilities. */
    for (y1 = 0;y1 < L;++y1) {
        for (y2 = 0;y2 < L;++y2) {
            for (y3 = 0;y3 < L;++y3) {
                floatval_t logp;
                
                ctx->labels[0] = y1;
                ctx->labels[1] = y2;
                ctx->labels[2] = y3;
                logp = crf1mc_logprob(ctx);

                fprintf(fp, "Check for the sequence %d-%d-%d... ", y1, y2, y3);
                if (fabs(scores[y1][y2][y3] / norm - exp(logp)) < eps) {
                    fprintf(fp, "OK (%f)\n", exp(logp));
                } else {
                    fprintf(fp, "FAIL: %f (%f)\n", exp(logp), scores[y1][y2][y3] / norm);
                }
            }
        }
    }

    /* Compute the marginal probability at t=0 */
    for (y1 = 0;y1 < L;++y1) {
        floatval_t a, b, c, p, q = 0.;
        for (y2 = 0;y2 < L;++y2) {
            for (y3 = 0;y3 < L;++y3) {
                q += scores[y1][y2][y3];
            }
        }
        q /= norm;

        a = FORWARD_SCORE_AT(ctx, 0)[y1];
        b = BACKWARD_SCORE_AT(ctx, 0)[y1];
        c = 1. / ctx->scale_factor[0];
        p = a * b * c;
        
        fprintf(fp, "Check for the marginal probability (0,%d)... ", y1);
        if (fabs(p - q) < eps) {
            fprintf(fp, "OK (%f)\n", p);
        } else {
            fprintf(fp, "FAIL: %f (%f)\n", p, q);
        }
    }

    /* Compute the marginal probability at t=1 */
    for (y2 = 0;y2 < L;++y2) {
        floatval_t a, b, c, p, q = 0.;
        for (y1 = 0;y1 < L;++y1) {
            for (y3 = 0;y3 < L;++y3) {
                q += scores[y1][y2][y3];
            }
        }
        q /= norm;

        a = FORWARD_SCORE_AT(ctx, 1)[y2];
        b = BACKWARD_SCORE_AT(ctx, 1)[y2];
        c = 1. / ctx->scale_factor[1];
        p = a * b * c;
        
        fprintf(fp, "Check for the marginal probability (1,%d)... ", y2);
        if (fabs(p - q) < eps) {
            fprintf(fp, "OK (%f)\n", p);
        } else {
            fprintf(fp, "FAIL: %f (%f)\n", p, q);
        }
    }

    /* Compute the marginal probability at t=2 */
    for (y3 = 0;y3 < L;++y3) {
        floatval_t a, b, c, p, q = 0.;
        for (y1 = 0;y1 < L;++y1) {
            for (y2 = 0;y2 < L;++y2) {
                q += scores[y1][y2][y3];
            }
        }
        q /= norm;

        a = FORWARD_SCORE_AT(ctx, 2)[y3];
        b = BACKWARD_SCORE_AT(ctx, 2)[y3];
        c = 1. / ctx->scale_factor[2];
        p = a * b * c;
        
        fprintf(fp, "Check for the marginal probability (2,%d)... ", y3);
        if (fabs(p - q) < eps) {
            fprintf(fp, "OK (%f)\n", p);
        } else {
            fprintf(fp, "FAIL: %f (%f)\n", p, q);
        }
    }

    /* Compute the marginal probabilities of transitions. */
    for (y1 = 0;y1 < L;++y1) {
        for (y2 = 0;y2 < L;++y2) {
            floatval_t a, b, s, t, p, q = 0.;
            for (y3 = 0;y3 < L;++y3) {
                q += scores[y1][y2][y3];
            }
            q /= norm;

            a = FORWARD_SCORE_AT(ctx, 0)[y1];
            b = BACKWARD_SCORE_AT(ctx, 1)[y2];
            s = STATE_SCORE_AT(ctx, 1)[y2];
            t = TRANS_SCORE_FROM(ctx, y1)[y2];
            p = a * t * s * b;

            fprintf(fp, "Check for the marginal probability (0,%d)-(1,%d)... ", y1, y2);
            if (fabs(p - q) < eps) {
                fprintf(fp, "OK (%f)\n", p);
            } else {
                fprintf(fp, "FAIL: %f (%f)\n", p, q);
            }
        }
    }
}

/*
 *      Linear-chain CRF training.
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
#include <string.h>

#include <crf.h>
#include "params.h"

#include "logging.h"
#include "crf1m.h"
#include <lbfgs.h>

typedef struct {
	float_t	feature_minfreq;
	int		feature_possible_states;
	int		feature_possible_transitions;
	int		feature_bos_eos;
	char*	regularization;
	float_t	regularization_sigma;
} crf1mt_option_t;

/**
 * First-order Markov CRF trainer.
 */
struct tag_crf1mt {
	int num_labels;			/**< Number of distinct output labels (L). */
	int num_attributes;		/**< Number of distinct attributes (A). */

	float_t sigma2inv;

	int max_items;

	crf_data_t* data;
	crf1m_context_t *ctx;	/**< CRF context. */

	logging_t* lg;

	void *cbe_instance;
	crf_evaluate_callback cbe_proc;

	feature_refs_t* attributes;
	feature_refs_t* forward_trans;
	feature_refs_t* backward_trans;
	feature_refs_t	bos_trans;
	feature_refs_t	eos_trans;

	int num_features;			/**< Number of distinct features (K). */

	/**
	 * Feature array.
	 *	Elements must be sorted by type, src, and dst in this order.
	 */
	crf1mt_feature_t *features;

	float_t *lambda;			/**< Array of lambda (feature weights) */
	float_t *best_lambda;
	int best;
	float_t *prob;

	crf_params_t* params;
	crf1mt_option_t opt;
};

#define	FEATURE(trainer, k) \
	(&(trainer)->features[(k)])
#define	ATTRIBUTE(trainer, a) \
	(&(trainer)->attributes[(a)])
#define	TRANSITION_FROM(trainer, i) \
	(&(trainer)->forward_trans[(i)])
#define	TRANSITION_TO(trainer, j) \
	(&(trainer)->backward_trans[(j)])
#define	TRANSITION_BOS(trainer) \
	(&(trainer)->bos_trans)
#define	TRANSITION_EOS(trainer) \
	(&(trainer)->eos_trans)

static void set_labels(crf1mt_t* trainer, const crf_instance_t* seq)
{
	int t;
	crf1m_context_t* ctx = trainer->ctx;
	const crf_item_t* item = NULL;
	const int T = seq->num_items;

	ctx->num_items = T;
	for (t = 0;t < T;++t) {
		item = &seq->items[t];
		ctx->labels[t] = seq->labels[t];
	}
}

static void state_score(crf1mt_t* trainer, const crf_instance_t* seq)
{
	int a, i, l, t, r;
	float_t scale, *state = NULL;
	crf1m_context_t* ctx = trainer->ctx;
	const crf_item_t* item = NULL;
	const crf1mt_feature_t* f = NULL;
	const feature_refs_t* attr = NULL;
	const int T = seq->num_items;
	const int L = trainer->num_labels;

	/* Loop over the items in the sequence. */
	for (t = 0;t < T;++t) {
		item = &seq->items[t];
		state = STATE_SCORE_AT(ctx, t);

		/* Initialize the state scores at position #t as zero. */
		for (i = 0;i < L;++i) {
			state[i] = 0;
		}

		/* Loop over the contents (attributes) attached to the item. */
		for (i = 0;i < item->num_contents;++i) {
			/* Access the list of state features associated with the attribute. */
			a = item->contents[i].aid;
			attr = ATTRIBUTE(trainer, a);
			/* A scale usually represents the atrribute frequency in the item. */
			scale = item->contents[i].scale;

			/* Loop over the state features associated with the attribute. */
			for (r = 0;r < attr->num_features;++r) {
				/* The state feature #(attr->fids[r]), which is represented by
				   the attribute #a, outputs the label #(f->dst). */
				f = FEATURE(trainer, attr->fids[r]);
				l = f->dst;
				state[l] += f->lambda * scale;
			}
		}
	}
}

static void transition_score(crf1mt_t* trainer)
{
	int i, j, r;
	float_t *trans = NULL;
	crf1m_context_t* ctx = trainer->ctx;
	const crf1mt_feature_t* f = NULL;
	const feature_refs_t* edge = NULL;
	const int L = trainer->num_labels;

	/* Initialize all transition scores as zero. */
	for (i = 0;i <= L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		for (j = 0;j <= L;++j) {
			trans[j] = 0;
		}
	}

	/* Compute transition scores from BOS to labels. */
	trans = TRANS_SCORE_FROM(ctx, L);
	edge = TRANSITION_BOS(trainer);
	for (r = 0;r < edge->num_features;++r) {
		/* Transition feature from BOS to #(f->dst). */
		f = FEATURE(trainer, edge->fids[r]);
		trans[f->dst] = f->lambda;
	}

	/* Compute transition scores between two labels. */
	for (i = 0;i < L;++i) {
		trans = TRANS_SCORE_FROM(ctx, i);
		edge = TRANSITION_FROM(trainer, i);
		for (r = 0;r < edge->num_features;++r) {
			/* Transition feature from #i to #(f->dst). */
			f = FEATURE(trainer, edge->fids[r]);
			trans[f->dst] = f->lambda;
		}		
	}

	/* Compute transition scores from labels to EOS. */
	edge = TRANSITION_EOS(trainer);
	for (r = 0;r < edge->num_features;++r) {
		/* Transition feature from #(f->src) to EOS. */
		f = FEATURE(trainer, edge->fids[r]);
		trans = TRANS_SCORE_FROM(ctx, f->src);
		trans[L] = f->lambda;
	}	
}

static void accumulate_expectation(crf1mt_t* trainer, const crf_instance_t* seq)
{
	int a, c, i, j, t, r;
	float_t scale, *prob = trainer->prob;
	crf1m_context_t* ctx = trainer->ctx;
	const float_t lognorm = ctx->log_norm;
	const float_t *fwd = NULL, *bwd = NULL, *state = NULL;
	crf1mt_feature_t* f = NULL;
	const feature_refs_t *attr = NULL, *trans = NULL;
	const crf_item_t* item = NULL;
	const int T = seq->num_items;
	const int L = trainer->num_labels;

	/*
		This code prioritizes the computation speed over the simplicity:
		it reduces the number of repeated calculations at the expense of
		its beauty. This routine calculates model expectations of features
		in four steps:

		(i)   state features at position #0 and BOS transition features
		(ii)  state features at position #T-1 and EOS transition features
		(iii) state features at position #t (0 < t < T-1)
		(iv)  transition features.
	 */

	/*
		(i) Calculate the probabilities at (0, i):
			p(0,i) = exp(fwd[0][i]) * exp(bwd[0][i]) / exp(lognorm)
		to compute expectations of transition features from BOS
		and state features at position #0.
	 */
	fwd = FORWARD_SCORE_AT(ctx, 0);
	bwd = BACKWARD_SCORE_AT(ctx, 0);
	for (i = 0;i < L;++i) {
		prob[i] = exp(fwd[i] + bwd[i] - lognorm);
	}

	/* Compute expectations for transition features from BOS. */
	trans = TRANSITION_BOS(trainer);
	for (r = 0;r < trans->num_features;++r) {
		f = FEATURE(trainer, trans->fids[r]);
		f->mexp += prob[f->dst];
	}

	/* Compute expectations for state features at position #0. */
	item = &seq->items[0];
	for (c = 0;c < item->num_contents;++c) {
		/* Access the attribute. */
		scale = item->contents[c].scale;
		a = item->contents[c].aid;
		attr = ATTRIBUTE(trainer, a);

		/* Loop over state features for the attribute. */
		for (r = 0;r < attr->num_features;++r) {
			/* Reuse the probability prob[f->dst]. */
			f = FEATURE(trainer, attr->fids[r]);
			f->mexp += (prob[f->dst] * scale);
		}
	}

	/*
		(ii) Calculate the probabilities at (T-1, i):
			p(T-1,i) = exp(fwd[T-1][i]) * exp(bwd[T-1][i]) / exp(lognorm)
		to compute expectations of transition features to EOS
		and state features at position #(T-1).
	 */
	fwd = FORWARD_SCORE_AT(ctx, T-1);
	bwd = BACKWARD_SCORE_AT(ctx, T-1);
	for (i = 0;i < L;++i) {
		prob[i] = exp(fwd[i] + bwd[i] - lognorm);
	}

	/* Compute expectations for transition features to EOS. */
	trans = TRANSITION_EOS(trainer);
	for (r = 0;r < trans->num_features;++r) {
		f = FEATURE(trainer, trans->fids[r]);
		f->mexp += prob[f->src];
	}

	/* Compute expectations for state features at position #(T-1). */
	item = &seq->items[T-1];
	for (c = 0;c < item->num_contents;++c) {
		/* Access the attribute. */
		scale = item->contents[c].scale;
		a = item->contents[c].aid;
		attr = ATTRIBUTE(trainer, a);

		/* Loop over state features for the attribute. */
		for (r = 0;r < attr->num_features;++r) {
			/* Reuse the probability prob[f->dst]. */
			f = FEATURE(trainer, attr->fids[r]);
			f->mexp += (prob[f->dst] * scale);
		}
	}

	/*
		(iii) For 0 < t < T-1, calculate the probabilities at (t, i):
			p(t,i) = exp(fwd[t][i]) * exp(bwd[t][i]) / exp(lognorm)
		to compute expectations of state features at position #t.
	 */
	for (t = 1;t < T-1;++t) {
		fwd = FORWARD_SCORE_AT(ctx, t);
		bwd = BACKWARD_SCORE_AT(ctx, t);

		/* Initialize the probabilities as -1, which means 'unfilled'.  */
		for (i = 0;i < L;++i) prob[i] = -1;

		/* Compute expectations for state features at position #t. */
		item = &seq->items[t];
		for (c = 0;c < item->num_contents;++c) {
			/* Access the attribute. */
			scale = item->contents[c].scale;
			a = item->contents[c].aid;
			attr = ATTRIBUTE(trainer, a);

			/* Loop over state features for the attribute. */
			for (r = 0;r < attr->num_features;++r) {
				f = FEATURE(trainer, attr->fids[r]);
				i = f->dst;

				/* Reuse the probability prob[i] if it has been calculated. */
				if (prob[i] == -1) {
					/* Unfilled: calculate the probability. */
					prob[i] = exp(fwd[i] + bwd[i] - lognorm);
				}
				f->mexp += (prob[i] * scale);
			}
		}
	}

	/*
		(iv) Calculate the probabilities of the path (t, i) -> (t+1, j)
			p(t+1,j|t,i)
				= exp(fwd[t][i]) * exp(lambda) * exp(state[t+1][j] * exp(bwd[t+1][j])
				/ exp(lognorm)
		to compute expectations of transition features.
	 */
	for (t = 0;t < T-1;++t) {
		fwd = FORWARD_SCORE_AT(ctx, t);
		state = STATE_SCORE_AT(ctx, t+1);
		bwd = BACKWARD_SCORE_AT(ctx, t+1);

		/* Loop over the labels (t, i) */
		for (i = 0;i < L;++i) {
			trans = TRANSITION_FROM(trainer, i);
			for (r = 0;r < trans->num_features;++r) {
				f = FEATURE(trainer, trans->fids[r]);
				j = f->dst;
				f->mexp += exp(fwd[i] + f->lambda + state[j] + bwd[j] - lognorm);
			}
		}
	}
}

static int init_feature_references(crf1mt_t* trainer, const int A, const int L)
{
	int i, k;
	feature_refs_t *fl = NULL;
	const int K = trainer->num_features;
	const crf1mt_feature_t* features = trainer->features;

	/*
		The purpose of this routine is to collect references (indices) of:
		- state features fired by each attribute (trainer->attributes)
		- transition features pointing from each label (trainer->forward_trans)
		- transition features pointing to each label (trainer->backward_trans)
		- BOS features (trainer->bos_trans)
		- EOS features (trainer->eos_trans).
	*/

	/* Initialize. */
	trainer->attributes = NULL;
	trainer->forward_trans = NULL;
	trainer->backward_trans = NULL;

	/* Allocate arrays for feature references. */
	trainer->attributes = (feature_refs_t*)calloc(A, sizeof(feature_refs_t));
	if (trainer->attributes == NULL) goto error_exit;
	trainer->forward_trans = (feature_refs_t*)calloc(L, sizeof(feature_refs_t));
	if (trainer->forward_trans == NULL) goto error_exit;
	trainer->backward_trans = (feature_refs_t*)calloc(L, sizeof(feature_refs_t));
	if (trainer->backward_trans == NULL) goto error_exit;
	memset(&trainer->bos_trans, 0, sizeof(feature_refs_t));
	memset(&trainer->eos_trans, 0, sizeof(feature_refs_t));

	/*
		Firstly, loop over the features to count the number of references.
		We don't want to use realloc() to avoid memory fragmentation.
	 */
	for (k = 0;k < K;++k) {
		const crf1mt_feature_t *f = &features[k];
		switch (f->type) {
		case FT_STATE:
			trainer->attributes[f->src].num_features++;
			break;
		case FT_TRANS:
			trainer->forward_trans[f->src].num_features++;
			trainer->backward_trans[f->dst].num_features++;
			break;
		case FT_TRANS_BOS:
			trainer->bos_trans.num_features++;
			break;
		case FT_TRANS_EOS:
			trainer->eos_trans.num_features++;
			break;
		}
	}

	/*
		Secondarily, allocate memory blocks to store the feature references.
		We also clear fl->num_features fields, which will be used to indicate
		the offset positions in the last phase.
	 */
	for (i = 0;i < trainer->num_attributes;++i) {
		fl = &trainer->attributes[i];
		fl->fids = (int*)calloc(fl->num_features, sizeof(int));
		if (fl->fids == NULL) goto error_exit;
		fl->num_features = 0;
	}
	for (i = 0;i < trainer->num_labels;++i) {
		fl = &trainer->forward_trans[i];
		fl->fids = (int*)calloc(fl->num_features, sizeof(int));
		if (fl->fids == NULL) goto error_exit;
		fl->num_features = 0;
		fl = &trainer->backward_trans[i];
		fl->fids = (int*)calloc(fl->num_features, sizeof(int));
		if (fl->fids == NULL) goto error_exit;
		fl->num_features = 0;
	}
	fl = &trainer->bos_trans;
	fl->fids = (int*)calloc(fl->num_features, sizeof(int));
	if (fl->fids == NULL) goto error_exit;
	fl->num_features = 0;
	fl = &trainer->eos_trans;
	fl->fids = (int*)calloc(fl->num_features, sizeof(int));
	if (fl->fids == NULL) goto error_exit;
	fl->num_features = 0;

	/*
		At last, store the feature indices.
	 */
	for (k = 0;k < K;++k) {
		const crf1mt_feature_t *f = &features[k];
		switch (f->type) {
		case FT_STATE:
			fl = &trainer->attributes[f->src];
			fl->fids[fl->num_features++] = k;
			break;
		case FT_TRANS:
			fl = &trainer->forward_trans[f->src];
			fl->fids[fl->num_features++] = k;
			fl = &trainer->backward_trans[f->dst];
			fl->fids[fl->num_features++] = k;
			break;
		case FT_TRANS_BOS:
			fl = &trainer->bos_trans;
			fl->fids[fl->num_features++] = k;
			break;
		case FT_TRANS_EOS:
			fl = &trainer->eos_trans;
			fl->fids[fl->num_features++] = k;
			break;
		}
	}

	return 0;

error_exit:
	if (trainer->attributes == NULL) {
		for (i = 0;i < A;++i) free(trainer->attributes[i].fids);
		free(trainer->attributes);
		trainer->attributes = NULL;
	}
	if (trainer->forward_trans == NULL) {
		for (i = 0;i < L;++i) free(trainer->forward_trans[i].fids);
		free(trainer->forward_trans);
		trainer->forward_trans = NULL;
	}
	if (trainer->backward_trans == NULL) {
		for (i = 0;i < L;++i) free(trainer->backward_trans[i].fids);
		free(trainer->backward_trans);
		trainer->backward_trans = NULL;
	}
	return -1;
}

int crf1mt_prepare(
	crf1mt_t* trainer,
	crf_data_t* data,
	crf1mt_features_t* features
	)
{
	int i;
	const int L = data->num_labels;
	const int A = data->num_attrs;
	const int T = data->max_item_length;

	/* Set basic parameters. */
	trainer->num_labels = L;
	trainer->num_attributes = A;

	/* Construct a CRF context. */
	trainer->ctx = crf1mc_new(L, T);

	/* Initialization for features and their weights. */
	trainer->features = features->features;
	trainer->num_features = features->num_features;
	trainer->lambda = (float_t*)calloc(trainer->num_features, sizeof(float_t));
	trainer->best_lambda = (float_t*)calloc(trainer->num_features, sizeof(float_t));
	trainer->best = 0;

	/* Allocate the work space for probability calculation. */
	trainer->prob = (float_t*)calloc(L, sizeof(float_t));

	/* Initialize the feature references. */
	init_feature_references(trainer, A, L);

	return 0;

error_exit:
	free(trainer->attributes);
	free(trainer->forward_trans);
	free(trainer->backward_trans);
	free(trainer->prob);
	return 0;
}

static int crf1mt_exchange_options(crf_params_t* params, crf1mt_option_t* opt, int mode)
{
	BEGIN_PARAM_MAP(params, mode)
		DDX_PARAM_FLOAT("feature.minfreq", opt->feature_minfreq, 0.0)
		DDX_PARAM_INT("feature.possible_states", opt->feature_possible_states, 0)
		DDX_PARAM_INT("feature.possible_transitions", opt->feature_possible_transitions, 0)
		DDX_PARAM_INT("feature.bos_eos", opt->feature_bos_eos, 1)
		DDX_PARAM_STRING("regularization", opt->regularization, "L2")
		DDX_PARAM_FLOAT("regularization.sigma", opt->regularization_sigma, 100.0)
	END_PARAM_MAP()

	return 0;
}

crf1mt_t* crf1mt_new()
{
	crf1mt_t* trainer = (crf1mt_t*)calloc(1, sizeof(crf1mt_t));
	trainer->lg = (logging_t*)calloc(1, sizeof(logging_t));

	/* Create an instance for CRF parameters. */
	trainer->params = params_create_instance();
	/* Set the default parameters. */
	crf1mt_exchange_options(trainer->params, &trainer->opt, 0);

	return trainer;
}

void crf1mt_delete(crf1mt_t* trainer)
{
	if (trainer != NULL) {
		free(trainer->lg);
	}
}

int crf_train_tag(crf_tagger_t* tagger, crf_instance_t *inst, crf_output_t* output)
{
	int i;
	float_t logprob = 0;
	crf1mt_t *crf1mt = (crf1mt_t*)tagger->internal;

	transition_score(crf1mt);
	set_labels(crf1mt, inst);
	state_score(crf1mt, inst);
	crf1mc_forward_score(crf1mt->ctx);
	crf1mc_backward_score(crf1mt->ctx);
	logprob = crf1mc_viterbi(crf1mt->ctx);
	logprob -= crf1mt->ctx->log_norm;

	output->probability = exp(logprob);
	for (i = 0;i < inst->num_items;++i) {
		output->labels[i] = crf1mt->ctx->labels[i];
	}
	output->num_labels = inst->num_items;

	return 0;
}



#if 1
static lbfgsfloat_t lbfgs_evaluate(void *instance, const lbfgsfloat_t *x, lbfgsfloat_t *g, const int n, const lbfgsfloat_t step)
{
	int i;
	float_t logp = 0, logl = 0, norm = 0;
	crf1mt_t* crf1mt = (crf1mt_t*)instance;
	crf_data_t* data = crf1mt->data;

	/*
		Set feature weights from the L-BFGS solver. Initialize model
		expectations as zero.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt_feature_t* f = &crf1mt->features[i];
		f->lambda = x[i];
		f->mexp = 0;
	}

	/*
		Set the scores (weights) of transition features here because
		these are independent of input label sequences.
	 */
	transition_score(crf1mt);

	/*
		Compute model expectations.
	 */
	for (i = 0;i < data->num_instances;++i) {
		/* Set label sequences and state scores. */
		set_labels(crf1mt, &data->instances[i]);
		state_score(crf1mt, &data->instances[i]);

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
		accumulate_expectation(crf1mt, &data->instances[i]);
	}

	/*
		Update the gradient vector.
	 */
	for (i = 0;i < crf1mt->num_features;++i) {
		const crf1mt_feature_t* f = &crf1mt->features[i];
		g[i] = -((f->oexp - f->mexp) - crf1mt->sigma2inv * f->lambda);
		norm += f->lambda * f->lambda;
		/*printf("%d (%f): %f -> %f, %f\n", i, f->lambda, f->mexp, f->oexp, g[i]);*/
	}

	logl -= (crf1mt->sigma2inv * norm * 0.5);
	return -logl;
}

static int lbfgs_progress(
	void *instance,
	const lbfgsfloat_t *x,
	const lbfgsfloat_t *g,
	const lbfgsfloat_t fx,
	const lbfgsfloat_t xnorm,
	const lbfgsfloat_t gnorm,
	const lbfgsfloat_t step,
	int n,
	int k,
	int ls)
{
	int i;
	crf1mt_t* crf1mt = (crf1mt_t*)instance;

	/* Set feature weights from the L-BFGS solver. */
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt_feature_t* f = &crf1mt->features[i];
		f->lambda = x[i];
		crf1mt->best_lambda[i] = x[i];
	}

	/* Report the progress. */
	logging(crf1mt->lg, "***** Iteration #%d *****\n", k);
	logging(crf1mt->lg, "Log-likelihood: %f\n", -fx);
	logging(crf1mt->lg, "Feature norm: %f\n", xnorm);
	logging(crf1mt->lg, "Error norm: %f\n", gnorm);
	logging(crf1mt->lg, "Line search trials: %d\n", ls);
	logging(crf1mt->lg, "Line search step: %f\n", step);

	/* Send the tagger with the current parameters. */
	if (crf1mt->cbe_proc != NULL) {
		int ret = 0;
		crf_tagger_t tagger;

		/* Construct a tagger instance. */
		tagger.internal = crf1mt;
		tagger.tag = crf_train_tag;

		/* Callback notification with the tagger object. */
		ret = crf1mt->cbe_proc(crf1mt->cbe_instance, &tagger);
	}

	logging(crf1mt->lg, "\n");
	return 0;
}
#endif


void crf_train_set_message_callback(crf_trainer_t* trainer, void *instance, crf_logging_callback cbm)
{
	crf1mt_t *crf1mt = (crf1mt_t*)trainer->internal;
	crf1mt->lg->func = cbm;
	crf1mt->lg->instance = instance;
}

void crf_train_set_evaluate_callback(crf_trainer_t* trainer, void *instance, crf_evaluate_callback cbe)
{
	crf1mt_t *crf1mt = (crf1mt_t*)trainer->internal;
	crf1mt->cbe_instance = instance;
	crf1mt->cbe_proc = cbe;
}

static int crf_train_train(crf_trainer_t* trainer, crf_data_t* data)
{
	int i;
	int ret = 0;
	float_t sigma = 10, *best_lambda = NULL;
	crf1mt_features_t* features = NULL;
	crf1mt_t *crf1mt = (crf1mt_t*)trainer->internal;
	crf_params_t *params = crf1mt->params;
	crf1mt_option_t *opt = &crf1mt->opt;

	/* Access parameters. */
	crf1mt_exchange_options(crf1mt->params, opt, -1);

	/* Report the parameters. */
	logging(crf1mt->lg, "Training first-order linear-chain CRFs (trainer.crf1m)\n");
	logging(crf1mt->lg, "feature.minfreq: %f\n", opt->feature_minfreq);
	logging(crf1mt->lg, "feature.possible_states: %d\n", opt->feature_possible_states);
	logging(crf1mt->lg, "feature.possible_transitions: %d\n", opt->feature_possible_transitions);
	logging(crf1mt->lg, "feature.bos_eos: %d\n", opt->feature_bos_eos);
	logging(crf1mt->lg, "regularization: %s\n", opt->regularization);
	logging(crf1mt->lg, "regularization.sigma: %f\n", opt->regularization_sigma);
	logging(crf1mt->lg, "Number of instances: %d\n", data->num_instances);
	logging(crf1mt->lg, "Number of distinct attributes: %d\n", data->num_attrs);
	logging(crf1mt->lg, "Number of distinct labels: %d\n", data->num_labels);
	logging(crf1mt->lg, "\n");

	/* Generate features. */
	logging(crf1mt->lg, "Feature generation\n");
	features = crf1mt_generate_features(
		data,
		opt->feature_possible_states ? 1 : 0,
		opt->feature_possible_transitions ? 1 : 0,
		opt->feature_minfreq,
		crf1mt->lg->func,
		crf1mt->lg->instance
		);
	logging(crf1mt->lg, "Number of features: %d\n", features->num_features);
	logging(crf1mt->lg, "\n");

	/* Preparation for training. */
	crf1mt_prepare(crf1mt, data, features);

	crf1mt->data = data;
	crf1mt->sigma2inv = 1.0 / (opt->regularization_sigma * opt->regularization_sigma);

	/* Call the L-BFGS solver. */
	logging(crf1mt->lg, "L-BFGS optimization\n");
	logging(crf1mt->lg, "\n");
	ret = lbfgs(crf1mt->num_features, crf1mt->lambda, lbfgs_evaluate, lbfgs_progress, crf1mt, NULL);
	if (ret == 0) {
		logging(crf1mt->lg, "L-BFGS resulted in convergence\n");
	} else {
		logging(crf1mt->lg, "L-BFGS terminated with error code (%d)\n", ret);
	}

	/* Store the feature weights. */
	best_lambda = ret == 0 ? crf1mt->lambda : crf1mt->best_lambda;
	for (i = 0;i < crf1mt->num_features;++i) {
		crf1mt_feature_t* f = &crf1mt->features[i];
		f->lambda = best_lambda[i];
	}

	return 0;
}

static int crf_train_save(crf_trainer_t* trainer, const char *filename, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
	crf1mt_t *crf1mt = (crf1mt_t*)trainer->internal;
	int a, k, l, ret;
	int *fmap = NULL, *amap = NULL;
	crf1mmw_t* writer = NULL;
	const feature_refs_t *edge = NULL, *attr = NULL;
	const float_t threshold = 0.01;
	const int L = crf1mt->num_labels;
	const int A = crf1mt->num_attributes;
	const int K = crf1mt->num_features;
	int J = 0, B = 0;

	/* Allocate and initialize the feature mapping. */
	fmap = (int*)calloc(K, sizeof(int));
	if (fmap == NULL) {
		goto error_exit;
	}
	for (k = 0;k < K;++k) fmap[k] = -1;

	/* Allocate and initialize the attribute mapping. */
	amap = (int*)calloc(A, sizeof(int));
	if (amap == NULL) {
		goto error_exit;
	}
	for (a = 0;a < A;++a) amap[a] = -1;

	/*
	 *	Open a model writer.
	 */
	writer = crf1mmw(filename);
	if (writer == NULL) {
		goto error_exit;
	}

	/* Open a feature chunk in the model file. */
	if (ret = crf1mmw_open_features(writer)) {
		goto error_exit;
	}

	/* Determine a set of active features and attributes. */
	for (k = 0, J = 0;k < crf1mt->num_features;++k) {
		crf1mt_feature_t* f = &crf1mt->features[k];
		if (f->lambda != 0) {
			/* The feature (f) will have a new feature id (#J). */
			fmap[k] = J++;		/* Feature #k -> #fmap[k]. */

			/* Write the feature. */
			if (ret = crf1mmw_put_feature(writer, fmap[k], f->lambda)) {
				goto error_exit;
			}

			if (f->type == FT_STATE) {
				a = f->src;
				if (amap[a] < 0) amap[a] = B++;	/* Attribute #a -> #amap[a]. */
			}
		}
	}

	/* Close the feature chunk. */
	if (ret = crf1mmw_close_features(writer)) {
		goto error_exit;
	}

	logging(crf1mt->lg, "Feature pruning %d -> %d\n", K, J);
	logging(crf1mt->lg, "Attribute pruning %d -> %d\n", A, B);

	/* Write labels. */
	logging(crf1mt->lg, "Writing %d labels\n", L);
	if (ret = crf1mmw_open_labels(writer, L)) {
		goto error_exit;
	}
	for (l = 0;l < L;++l) {
		char *str = NULL;
		labels->to_string(labels, l, &str);
		if (str != NULL) {
			if (ret = crf1mmw_put_label(writer, l, str)) {
				goto error_exit;
			}
			labels->free(labels, str);
		}
	}
	if (ret = crf1mmw_close_labels(writer)) {
		goto error_exit;
	}

	/* Write attributes. */
	logging(crf1mt->lg, "Writing %d attributes\n", B);
	if (ret = crf1mmw_open_attrs(writer, B)) {
		goto error_exit;
	}
	for (a = 0;a < A;++a) {
		if (0 <= amap[a]) {
			char *str = NULL;
			attrs->to_string(attrs, a, &str);
			if (str != NULL) {
				if (ret = crf1mmw_put_attr(writer, amap[a], str)) {
					goto error_exit;
				}
				attrs->free(attrs, str);
			}
		}
	}
	if (ret = crf1mmw_close_attrs(writer)) {
		goto error_exit;
	}

	/* Write label feature references. */
	logging(crf1mt->lg, "Writing label feature references\n");
	if (ret = crf1mmw_open_labelrefs(writer, L+2)) {
		goto error_exit;
	}
	for (l = 0;l < L;++l) {
		edge = TRANSITION_FROM(crf1mt, l);
		if (ret = crf1mmw_put_labelref(writer, l, edge, fmap)) {
			goto error_exit;
		}
	}
	edge = TRANSITION_BOS(crf1mt);
	if (ret = crf1mmw_put_labelref(writer, L, edge, fmap)) {
		goto error_exit;
	}
	edge = TRANSITION_EOS(crf1mt);
	if (ret = crf1mmw_put_labelref(writer, L+1, edge, fmap)) {
		goto error_exit;
	}
	if (ret = crf1mmw_close_labelrefs(writer)) {
		goto error_exit;
	}

	/* Write attribute feature references. */
	logging(crf1mt->lg, "Writing attribute feature references\n");
	if (ret = crf1mmw_open_attrrefs(writer, B)) {
		goto error_exit;
	}
	for (a = 0;a < A;++a) {
		if (0 <= amap[a]) {
			attr = ATTRIBUTE(crf1mt, amap[a]);
			if (ret = crf1mmw_put_attrref(writer, amap[a], attr, fmap)) {
				goto error_exit;
			}
		}
	}
	if (ret = crf1mmw_close_attrrefs(writer)) {
		goto error_exit;
	}

	crf1mmw_close(writer);
	free(amap);
	free(fmap);
	return 0;

error_exit:
	if (writer != NULL) {
		crf1mmw_close(writer);
	}
	if (amap != NULL) {
		free(amap);
	}
	if (fmap != NULL) {
		free(fmap);
	}
	return ret;
}

static int crf_train_addref(crf_trainer_t* trainer)
{
	return crf_interlocked_increment(&trainer->refcount);
}

static int crf_train_release(crf_trainer_t* trainer)
{
	int count = crf_interlocked_decrement(&trainer->refcount);
	if (count == 0) {
	}
	return count;
}

static crf_params_t* crf_train_params(crf_trainer_t* trainer)
{
	crf1mt_t *crf1mt = (crf1mt_t*)trainer->internal;
	crf_params_t* params = crf1mt->params;
	params->addref(params);
	return params;
}


int crf1mt_create_instance(const char *interface, void **ptr)
{
	if (strcmp(interface, "trainer.crf1m") == 0) {
		crf_trainer_t* trainer = (crf_trainer_t*)calloc(1, sizeof(crf_trainer_t));

		trainer->refcount = 1;
		trainer->addref = crf_train_addref;
		trainer->release = crf_train_release;

		trainer->params = crf_train_params;
	
		trainer->set_message_callback = crf_train_set_message_callback;
		trainer->set_evaluate_callback = crf_train_set_evaluate_callback;
		trainer->trainer = crf_train_train;
		trainer->save = crf_train_save;
		trainer->internal = crf1mt_new();

		*ptr = trainer;
		return 0;
	} else {
		return 1;
	}
}

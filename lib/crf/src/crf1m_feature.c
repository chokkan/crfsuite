/*
 *      Feature generation for linear-chain CRF.
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

/* $Id:$ */


#ifdef	HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>

#include "logging.h"
#include "crf1m.h"
#include "rumavl.h"	/* AVL tree library necessary for feature generation. */

/**
 * Feature set.
 */
typedef struct {
	RUMAVL* avl;	/**< Root node of the AVL tree. */
	int num;		/**< Number of features in the AVL tree. */
} featureset_t;


#define	COMP(a, b)	((a)>(b))-((a)<(b))

static int featureset_comp(const void *x, const void *y, size_t n)
{
	int ret = 0;
	const crf1mt_feature_t* f1 = (const crf1mt_feature_t*)x;
	const crf1mt_feature_t* f2 = (const crf1mt_feature_t*)y;

	ret = COMP(f1->type, f2->type);
	if (ret == 0) {
		ret = COMP(f1->src, f2->src);
		if (ret == 0) {
			ret = COMP(f1->dst, f2->dst);
		}
	}
	return ret;
}

static featureset_t* featureset_new()
{
	featureset_t* set = NULL;
	set = (featureset_t*)calloc(1, sizeof(featureset_t));
	if (set != NULL) {
		set->num = 0;
		set->avl = rumavl_new(sizeof(crf1mt_feature_t), featureset_comp);
		if (set->avl == NULL) {
			free(set);
			set = NULL;
		}
	}
	return set;
}

static void featureset_delete(featureset_t* set)
{
	if (set != NULL) {
		rumavl_destroy(set->avl);
		free(set);
	}
}

static int featureset_add(featureset_t* set, const crf1mt_feature_t* f)
{
	/* Check whether if the feature already exists. */
	crf1mt_feature_t *p = (crf1mt_feature_t*)rumavl_find(set->avl, f);
	if (p == NULL) {
		/* Insert the feature to the feature set. */
		rumavl_insert(set->avl, f);
		++set->num;
	} else {
		/* An existing feature: add the observation expectation. */
		p->oexp += f->oexp;
	}
	return 0;
}

static void featureset_generate(crf1mt_features_t* features, featureset_t* set, float_t minfreq)
{
	int n = 0, k = 0;
	RUMAVL_NODE *node = NULL;
	crf1mt_feature_t *f = NULL;

	features->features = 0;

	/* The first pass: count the number of valid features. */
	while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
		if (minfreq <= f->oexp) {
			++n;
		}
	}

	/* The second path: copy the valid features to the feature array. */
	features->features = (crf1mt_feature_t*)calloc(n, sizeof(crf1mt_feature_t));
	if (features->features != NULL) {
		node = NULL;
		while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
			if (minfreq <= f->oexp) {
				memcpy(&features->features[k], f, sizeof(crf1mt_feature_t));
				++k;
			}
		}
		features->num_features = n;
	}
}

crf1mt_features_t* crf1mt_generate_features(
	const crf_data_t *data,
	int connect_all_attrs,
	int connect_all_edges,
	float_t minfreq,
	crf_logging_callback func,
	void *instance
	)
{
	int c, i, j, s, t;
	crf1mt_feature_t f;
	featureset_t* set = NULL;
	crf1mt_features_t *features = NULL;
	const int L = data->num_labels;
	logging_t lg;

	lg.func = func;
	lg.instance = instance;
	lg.percent = 0;

	/* Allocate a feature container. */
	features = (crf1mt_features_t*)calloc(1, sizeof(crf1mt_features_t));

	/* Create an instance of feature set. */
	set = featureset_new();

	/* Loop over the sequences in the training data. */
	logging_progress_start(&lg);

	for (s = 0;s < data->num_instances;++s) {
		int prev = L, cur = 0;
		const crf_item_t* item = NULL;
		const crf_instance_t* seq = &data->instances[s];
		const int T = seq->num_items;

		/* Loop over the items in the sequence. */
		for (t = 0;t < T;++t) {
			item = &seq->items[t];
			cur = seq->labels[t];

			/* Transition feature: label #prev -> label #(item->yid).
			   Features with previous label #L are transition BOS. */
			f.type = (prev == L) ? FT_TRANS_BOS : FT_TRANS;
			f.src = prev;
			f.dst = cur;
			f.oexp = 1;
			f.mexp = 0;
			f.lambda = 0;
			featureset_add(set, &f);

			for (c = 0;c < item->num_contents;++c) {
				/* State feature: attribute #a -> state #(item->yid). */
				f.type = FT_STATE;
				f.src = item->contents[c].aid;
				f.dst = cur;
				f.oexp = item->contents[c].scale;
				f.mexp = 0;
				f.lambda = 0;
				featureset_add(set, &f);

				/* Generate state features connecting attributes with all
				   output labels. These features are not unobserved in the
				   training data (zero expexcations). */
				if (connect_all_attrs) {
					for (i = 0;i < L;++i) {
						f.type = FT_STATE;
						f.src = item->contents[c].aid;
						f.dst = i;
						f.oexp = 0;
						f.mexp = 0;
						f.lambda = 0;
						featureset_add(set, &f);
					}
				}
			}

			prev = cur;
		}

		/* Transition EOS feature: label #(item->yid) -> EOS. */
		item = &seq->items[T-1];
		f.type = FT_TRANS_EOS;
		f.src = cur;
		f.dst = L;
		f.oexp = 1;
		f.mexp = 0;
		f.lambda = 0;
		featureset_add(set, &f);

		logging_progress(&lg, s * 100 / data->num_instances);
	}
	logging_progress_end(&lg);

	/* Make sure to generate all possible BOS and EOS features. */
	for (i = 0;i < L;++i) {
		f.type = FT_TRANS_BOS;
		f.src = L;
		f.dst = i;
		f.oexp = 0;
		f.mexp = 0;
		f.lambda = 0;
		featureset_add(set, &f);

		f.type = FT_TRANS_EOS;
		f.src = i;
		f.dst = L;
		f.oexp = 0;
		f.mexp = 0;
		f.lambda = 0;
		featureset_add(set, &f);
	}

	/* Generate edge features representing all pairs of labels.
	   These features are not unobserved in the training data
	   (zero expexcations). */
	if (connect_all_edges) {
		for (i = 0;i < L;++i) {
			for (j = 0;j < L;++j) {
				f.type = FT_TRANS;
				f.src = i;
				f.dst = j;
				f.oexp = 0;
				f.mexp = 0;
				f.lambda = 0;
				featureset_add(set, &f);
			}
		}
	}

	/* Convert the feature set to an feature array. */
	featureset_generate(features, set, minfreq);

	/* Delete the feature set. */
	featureset_delete(set);

	return features;
}

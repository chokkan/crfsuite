/*
 *      Linear-chain CRF tagger.
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>

#include "crf1m.h"

static int attrs_get(crf_dictionary_t* dic, const char *str)
{
	return -1;	/* Not supported. */
}

static int attrs_to_id(crf_dictionary_t* dic, const char *str)
{
	crf1mm_t *crf1mm = (crf1mm_t*)dic->internal;
	return crf1mm_to_aid(crf1mm, str);
}

static int attrs_to_string(crf_dictionary_t* dic, int id, char **pstr)
{
	return -1;	/* Not supported. */
}

static int attrs_num(crf_dictionary_t* dic)
{
	crf1mm_t *crf1mm = (crf1mm_t*)dic->internal;
	return crf1mm_get_num_attrs(crf1mm);
}

static void attrs_free(crf_dictionary_t* dic, char *str)
{
	/* Not supported. */
}

static int attrs_release(crf_dictionary_t* dic)
{
	return 0;
}



static int labels_get(crf_dictionary_t* dic, const char *str)
{
	return -1;	/* Not supported. */
}

static int labels_to_id(crf_dictionary_t* dic, const char *str)
{
	crf1mm_t *crf1mm = (crf1mm_t*)dic->internal;
	return crf1mm_to_lid(crf1mm, str);
}

static int labels_to_string(crf_dictionary_t* dic, int id, char **pstr)
{
	crf1mm_t *crf1mm = (crf1mm_t*)dic->internal;
	const char *str = crf1mm_to_label(crf1mm, id);
	*pstr = str;
	return 0;
}

static int labels_num(crf_dictionary_t* dic)
{
	crf1mm_t *crf1mm = (crf1mm_t*)dic->internal;
	return crf1mm_get_num_labels(crf1mm);
}

static void labels_free(crf_dictionary_t* dic, char *str)
{
	/* Unnecessary. */
}

static int labels_release(crf_dictionary_t* dic)
{
	return 0;
}




int crf_create_tagger(
	const char *filename,
	crf_tagger_t** ptr_tagger,
	crf_dictionary_t** ptr_attrs,
	crf_dictionary_t** ptr_labels
	)
{
	crf1mm_t* model = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;

	model = crf1mm(filename);

	attrs = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
	if (attrs == NULL) {
	}
	attrs->internal = model;
	attrs->get = attrs_get;
	attrs->to_id = attrs_to_id;
	attrs->to_string = attrs_to_string;
	attrs->num = attrs_num;
	attrs->free = attrs_free;
	attrs->release = attrs_release;
	*ptr_attrs = attrs;

	labels = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
	if (labels == NULL) {
	}
	labels->internal = model;
	labels->get = labels_get;
	labels->to_id = labels_to_id;
	labels->to_string = labels_to_string;
	labels->num = labels_num;
	labels->free = labels_free;
	labels->release = labels_release;
	*ptr_labels = labels;

	return 0;
}


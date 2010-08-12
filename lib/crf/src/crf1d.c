/*
 *      The 1st-order linear-chain CRF with dyad features (CRF1d).
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>
#include "params.h"

#include "logging.h"
#include "crf1d.h"

int crf1m_model_create(const char *filename, crf_model_t** ptr_model);

int crf1m_create_instance_from_file(const char *filename, void **ptr)
{
    return crf1m_model_create(filename, (crf_model_t**)ptr);
}


typedef struct {
    crf1dm_t*    crf1dm;

    crf_dictionary_t*    attrs;
    crf_dictionary_t*    labels;
    crf_tagger_t*        tagger;
} model_internal_t;

/*
 *    Implementation of crf_dictionary_t object representing attributes.
 *    This object is instantiated only by a crf_model_t object.
 */

static int model_attrs_addref(crf_dictionary_t* dic)
{
    return crf_interlocked_increment(&dic->nref);
}

static int model_attrs_release(crf_dictionary_t* dic)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&dic->nref);
}

static int model_attrs_get(crf_dictionary_t* dic, const char *str)
{
    return CRFERR_NOTSUPPORTED;    /* This object is ready only. */
}

static int model_attrs_to_id(crf_dictionary_t* dic, const char *str)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    return crf1dm_to_aid(crf1dm, str);
}

static int model_attrs_to_string(crf_dictionary_t* dic, int id, char const **pstr)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    const char *str = crf1dm_to_attr(crf1dm, id);
    *pstr = str;
    return 0;
}

static int model_attrs_num(crf_dictionary_t* dic)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    return crf1dm_get_num_attrs(crf1dm);
}

static void model_attrs_free(crf_dictionary_t* dic, const char *str)
{
    /* Unnecessary: all strings are freed on the final release. */
}




/*
 *    Implementation of crf_dictionary_t object representing labels.
 *    This object is instantiated only by a crf_model_t object.
 */

static int model_labels_addref(crf_dictionary_t* dic)
{
    return crf_interlocked_increment(&dic->nref);
}

static int model_labels_release(crf_dictionary_t* dic)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&dic->nref);
}

static int model_labels_get(crf_dictionary_t* dic, const char *str)
{
    return CRFERR_NOTSUPPORTED;    /* This object is ready only. */
}

static int model_labels_to_id(crf_dictionary_t* dic, const char *str)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    return crf1dm_to_lid(crf1dm, str);
}

static int model_labels_to_string(crf_dictionary_t* dic, int id, char const **pstr)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    const char *str = crf1dm_to_label(crf1dm, id);
    *pstr = str;
    return 0;
}

static int model_labels_num(crf_dictionary_t* dic)
{
    crf1dm_t *crf1dm = (crf1dm_t*)dic->internal;
    return crf1dm_get_num_labels(crf1dm);
}

static void model_labels_free(crf_dictionary_t* dic, const char *str)
{
    /* Unnecessary: all strings are freed on the final release. */
}



static int tagger_addref(crf_tagger_t* tagger)
{
    return crf_interlocked_increment(&tagger->nref);
}

static int tagger_release(crf_tagger_t* tagger)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&tagger->nref);
}

static int tagger_tag(crf_tagger_t* tagger, crf_sequence_t *inst, crf_output_t* output)
{
    crf1dt_t* crf1dt = (crf1dt_t*)tagger->internal;
    crf1dt_tag(crf1dt, inst, output);
    return 0;
}




/*
 *    Implementation of crf_model_t object.
 *    This object is instantiated by crf1m_model_create() function.
 */

static int model_addref(crf_model_t* model)
{
    return crf_interlocked_increment(&model->nref);
}

static int model_release(crf_model_t* model)
{
    int count = crf_interlocked_decrement(&model->nref);
    if (count == 0) {
        /* This instance is being destroyed. */
        model_internal_t* internal = (model_internal_t*)model->internal;
        crf1dt_delete((crf1dt_t*)internal->tagger->internal);
        free(internal->tagger);
        free(internal->labels);
        free(internal->attrs);
        crf1dm_close(internal->crf1dm);
        free(internal);
        free(model);
    }
    return count;
}

static int model_get_tagger(crf_model_t* model, crf_tagger_t** ptr_tagger)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_tagger_t* tagger = internal->tagger;

    tagger->addref(tagger);
    *ptr_tagger = tagger;
    return 0;
}

static int model_get_labels(crf_model_t* model, crf_dictionary_t** ptr_labels)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_dictionary_t* labels = internal->labels;

    labels->addref(labels);
    *ptr_labels = labels;
    return 0;
}

static int model_get_attrs(crf_model_t* model, crf_dictionary_t** ptr_attrs)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_dictionary_t* attrs = internal->attrs;

    attrs->addref(attrs);
    *ptr_attrs = attrs;
    return 0;
}

static int model_dump(crf_model_t* model, FILE *fpo)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf1dm_dump(internal->crf1dm, fpo);
    return 0;
}

int crf1m_model_create(const char *filename, crf_model_t** ptr_model)
{
    int ret = 0;
    crf1dm_t *crf1dm = NULL;
    crf1dt_t *crf1dt = NULL;
    crf_model_t *model = NULL;
    model_internal_t *internal = NULL;
    crf_tagger_t *tagger = NULL;
    crf_dictionary_t *attrs = NULL, *labels = NULL;

    *ptr_model = NULL;

    /* Open the model file. */
    crf1dm = crf1dm_new(filename);
    if (crf1dm == NULL) {
        ret = CRFERR_INCOMPATIBLE;
        goto error_exit;
    }

    /* Construct a tagger based on the model. */
    crf1dt = crf1dt_new(crf1dm);
    if (crf1dt == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Create an instance of internal data attached to the model. */
    internal = (model_internal_t*)calloc(1, sizeof(model_internal_t));
    if (internal == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Create an instance of dictionary object for attributes. */
    attrs = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
    if (attrs == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    attrs->internal = crf1dm;
    attrs->addref = model_attrs_addref;
    attrs->release = model_attrs_release;
    attrs->get = model_attrs_get;
    attrs->to_id = model_attrs_to_id;
    attrs->to_string = model_attrs_to_string;
    attrs->num = model_attrs_num;
    attrs->free = model_attrs_free;

    /* Create an instance of dictionary object for labels. */
    labels = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
    if (labels == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    labels->internal = crf1dm;
    labels->addref = model_labels_addref;
    labels->release = model_labels_release;
    labels->get = model_labels_get;
    labels->to_id = model_labels_to_id;
    labels->to_string = model_labels_to_string;
    labels->num = model_labels_num;
    labels->free = model_labels_free;

    /* */
    /* Create an instance of tagger object. */
    tagger = (crf_tagger_t*)calloc(1, sizeof(crf_tagger_t));
    if (tagger == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    tagger->internal = crf1dt;
    tagger->addref = tagger_addref;
    tagger->release = tagger_release;
    tagger->tag = tagger_tag;

    /* Set the internal data. */
    internal->crf1dm = crf1dm;
    internal->attrs = attrs;
    internal->labels = labels;
    internal->tagger = tagger;

    /* Create an instance of model object. */
    model = (crf_model_t*)calloc(1, sizeof(crf_model_t));
    if (model == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    model->internal = internal;
    model->nref = 1;
    model->addref = model_addref;
    model->release = model_release;
    model->get_attrs = model_get_attrs;
    model->get_labels = model_get_labels;
    model->get_tagger = model_get_tagger;
    model->dump = model_dump;

    *ptr_model = model;
    return 0;

error_exit:
    free(tagger);
    free(labels);
    free(attrs);
    if (crf1dt != NULL) {
        crf1dt_delete(crf1dt);
    }
    if (crf1dm != NULL) {
        crf1dm_close(crf1dm);
    }
    free(internal);
    free(model);
    return ret;
}

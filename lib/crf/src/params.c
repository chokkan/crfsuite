/*
 *      Parameter exchange.
 *
 * Copyright (c) 2007-2009, Naoaki Okazaki
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

#include <os.h>

#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "quark.h"

enum {
    PT_NONE = 0,
    PT_INT,
    PT_FLOAT,
    PT_STRING,
};

typedef struct {
    char*    name;
    int        type;
    int        val_i;
    floatval_t val_f;
    char*    val_s;
} param_t;

typedef struct {
    int num_params;
    param_t* params;
} params_t;

static char *mystrdup(const char *src)
{
    char *dst = (char*)malloc(strlen(src) + 1);
    if (dst != NULL) {
        strcpy(dst, src);
    }
    return dst;
}

static int params_addref(crf_params_t* params)
{
    return crf_interlocked_increment(&params->nref);
}

static int params_release(crf_params_t* params)
{
    int count = crf_interlocked_decrement(&params->nref);
    if (count == 0) {
        int i;
        params_t* pars = (params_t*)params->internal;
        for (i = 0;i < pars->num_params;++i) {
            free(pars->params[i].name);
            free(pars->params[i].val_s);
        }
        free(pars);
    }
    return count;
}

static param_t* find_param(params_t* pars, const char *name)
{
    int i;

    for (i = 0;i < pars->num_params;++i) {
        if (strcmp(pars->params[i].name, name) == 0) {
            return &pars->params[i];
        }
    }

    return NULL;
}

static int params_set(crf_params_t* params, const char *name, const char *value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    switch (par->type) {
    case PT_INT:
        par->val_i = (value != NULL) ? atoi(value) : 0;
        break;
    case PT_FLOAT:
        par->val_f = (value != NULL) ? (floatval_t)atof(value) : 0;
        break;
    case PT_STRING:
        free(par->val_s);
        par->val_s = (value != NULL) ? mystrdup(value) : mystrdup("");
    }
    return 0;
}

static int params_set_int(crf_params_t* params, const char *name, int value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_INT) return -1;
    par->val_i = value;
    return 0;
}

static int params_set_float(crf_params_t* params, const char *name, floatval_t value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_FLOAT) return -1;
    par->val_f = value;
    return 0;
}

static int params_set_string(crf_params_t* params, const char *name, const char *value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_STRING) return -1;
    free(par->val_s);
    par->val_s = mystrdup(value);
    return 0;
}

static int params_get_int(crf_params_t* params, const char *name, int *value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_INT) return -1;
    *value = par->val_i;
    return 0;
}

static int params_get_float(crf_params_t* params, const char *name, floatval_t *value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_FLOAT) return -1;
    *value = par->val_f;
    return 0;
}

static int params_get_string(crf_params_t* params, const char *name, char **value)
{
    params_t* pars = (params_t*)params->internal;
    param_t* par = find_param(pars, name);
    if (par == NULL) return -1;
    if (par->type != PT_STRING) return -1;
    *value = par->val_s;
    return 0;
}

crf_params_t* params_create_instance()
{
    crf_params_t* params = (crf_params_t*)calloc(1, sizeof(crf_params_t));

    if (params != NULL) {
        /* Construct the internal data. */
        params->internal = (params_t*)calloc(1, sizeof(params_t));
        if (params->internal == NULL) {
            free(params);
            return NULL;
        }

        /* Set member functions. */
        params->nref = 1;
        params->addref = params_addref;
        params->release = params_release;
        params->set = params_set;
        params->set_int = params_set_int;
        params->set_float = params_set_float;
        params->set_string = params_set_string;
        params->get_int = params_get_int;
        params->get_float = params_get_float;
        params->get_string = params_get_string;
    }

    return params;
}

int params_add_int(crf_params_t* params, const char *name, int value)
{
    param_t* par = NULL;
    params_t* pars = (params_t*)params->internal;
    pars->params = (param_t*)realloc(pars->params, (pars->num_params+1) * sizeof(param_t));
    if (pars->params == NULL) {
        return -1;
    }

    par = &pars->params[pars->num_params++];
    memset(par, 0, sizeof(*par));
    par->name = mystrdup(name);
    par->type = PT_INT;
    par->val_i = value;
    return 0;
}

int params_add_float(crf_params_t* params, const char *name, floatval_t value)
{
    param_t* par = NULL;
    params_t* pars = (params_t*)params->internal;
    pars->params = (param_t*)realloc(pars->params, (pars->num_params+1) * sizeof(param_t));
    if (pars->params == NULL) {
        return -1;
    }

    par = &pars->params[pars->num_params++];
    memset(par, 0, sizeof(*par));
    par->name = mystrdup(name);
    par->type = PT_FLOAT;
    par->val_f = value;
    return 0;
}

int params_add_string(crf_params_t* params, const char *name, const char *value)
{
    param_t* par = NULL;
    params_t* pars = (params_t*)params->internal;
    pars->params = (param_t*)realloc(pars->params, (pars->num_params+1) * sizeof(param_t));
    if (pars->params == NULL) {
        return -1;
    }

    par = &pars->params[pars->num_params++];
    memset(par, 0, sizeof(*par));
    par->name = mystrdup(name);
    par->type = PT_STRING;
    par->val_s = mystrdup(value);
    return 0;
}

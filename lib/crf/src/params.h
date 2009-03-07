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

#ifndef    __PARAMS_H__
#define    __PARAMS_H__

crf_params_t* params_create_instance();
int params_add_int(crf_params_t* params, const char *name, int value);
int params_add_float(crf_params_t* params, const char *name, floatval_t value);
int params_add_string(crf_params_t* params, const char *name, const char *value);

#define    BEGIN_PARAM_MAP(params, mode) \
    do { \
        int __ret = 0; \
        int __mode = mode; \
        crf_params_t* __params = params;

#define    END_PARAM_MAP() \
    } while (0) ;

#define    DDX_PARAM_INT(name, var, defval, help) \
    if (__mode < 0) \
        __ret = __params->get_int(__params, name, &var); \
    else if (__mode > 0) \
        __ret = __params->set_int(__params, name, var); \
    else \
        __ret = params_add_int(__params, name, defval);

#define    DDX_PARAM_FLOAT(name, var, defval, help) \
    if (__mode < 0) \
        __ret = __params->get_float(__params, name, &var); \
    else if (__mode > 0) \
        __ret = __params->set_float(__params, name, var); \
    else \
        __ret = params_add_float(__params, name, defval);

#define    DDX_PARAM_STRING(name, var, defval, help) \
    if (__mode < 0) \
        __ret = __params->get_string(__params, name, &var); \
    else if (__mode > 0) \
        __ret = __params->set_string(__params, name, var); \
    else \
        __ret = params_add_string(__params, name, defval);

#endif/*__PARAMS_H__*/

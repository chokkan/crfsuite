/*
 *      Mathematical operations for vectors.
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

#ifndef __VECMATH_H__
#define __VECMATH_H__

#include <math.h>
#include <memory.h>

inline static void veczero(floatval_t *x, const int n)
{
    memset(x, 0, sizeof(floatval_t) * n);
}

inline static void vecset(floatval_t *x, const floatval_t a, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        x[i] = a;
    }
}

inline static void veccopy(floatval_t *y, const floatval_t *x, const int n)
{
    memcpy(y, x, sizeof(floatval_t) * n);
}

inline static void vecadd(floatval_t *y, const floatval_t *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] += x[i];
    }
}

inline static void vecaadd(floatval_t *y, const floatval_t a, const floatval_t *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] += a * x[i];
    }
}

inline static void vecsub(floatval_t *y, const floatval_t *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] -= x[i];
    }
}

inline static void vecasub(floatval_t *y, const floatval_t a, const floatval_t *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] -= a * x[i];
    }
}

inline static void vecmul(floatval_t *y, const floatval_t *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] *= x[i];
    }
}

inline static void vecinv(floatval_t *y, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] = 1. / y[i];
    }
}

inline static void vecscale(floatval_t *y, const floatval_t a, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] *= a;
    }
}

inline static floatval_t vecdot(const floatval_t *x, const floatval_t *y, const int n)
{
    int i;
    floatval_t s = 0;
    for (i = 0;i < n;++i) {
        s += x[i] * y[i];
    }
    return s;
}

inline static floatval_t vecsum(floatval_t* x, const int n)
{
    int i;
    floatval_t s = 0.;

    for (i = 0;i < n;++i) {
        s += x[i];
    }
    return s;
}

inline static void vecexp(floatval_t* x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        x[i] = (x[i] == 0. ? 1. : exp(x[i]));
    }
}

inline static floatval_t vecsumlog(floatval_t* x, const int n)
{
    int i;
    floatval_t s = 0.;
    for (i = 0;i < n;++i) {
        s += log(x[i]);
    }
    return s;
}

#endif/*__VECMATH_H__*/

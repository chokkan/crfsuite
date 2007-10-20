/*
 *      ANSI C implementation of vector operations.
 *
 * Copyright (c) 2007, Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id:$ */

#include <stdlib.h>
#include <memory.h>

#if		LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
#define	fsigndiff(x, y)	(((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
#else
#define	fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
#endif/*LBFGS_IEEE_FLOAT*/

inline static void* vecalloc(size_t size)
{
	void *memblock = malloc(size);
	if (memblock) {
		memset(memblock, 0, size);
	}
	return memblock;
}

inline static void vecfree(void *memblock)
{
	free(memblock);
}

inline static void vecset(lbfgsfloat_t *x, const lbfgsfloat_t c, const int n)
{
	int i;
	
	for (i = 0;i < n;++i) {
		x[i] = c;
	}
}

inline static void veccpy(lbfgsfloat_t *y, const lbfgsfloat_t *x, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		y[i] = x[i];
	}
}

inline static void vecncpy(lbfgsfloat_t *y, const lbfgsfloat_t *x, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		y[i] = -x[i];
	}
}

inline static void vecadd(lbfgsfloat_t *y, const lbfgsfloat_t *x, const lbfgsfloat_t c, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		y[i] += c * x[i];
	}
}

inline static void vecdiff(lbfgsfloat_t *z, const lbfgsfloat_t *x, const lbfgsfloat_t *y, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		z[i] = x[i] - y[i];
	}
}

inline static void vecscale(lbfgsfloat_t *y, const lbfgsfloat_t c, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		y[i] *= c;
	}
}

inline static void vecmul(lbfgsfloat_t *y, const lbfgsfloat_t *x, const int n)
{
	int i;

	for (i = 0;i < n;++i) {
		y[i] *= x[i];
	}
}

inline static void vecdot(lbfgsfloat_t* s, const lbfgsfloat_t *x, const lbfgsfloat_t *y, const int n)
{
	int i;
	*s = 0.;
	for (i = 0;i < n;++i) {
		*s += x[i] * y[i];
	}
}

inline static void vecnorm(lbfgsfloat_t* s, const lbfgsfloat_t *x, const int n)
{
	vecdot(s, x, x, n);
	*s = (lbfgsfloat_t)sqrt(*s);
}

inline static void vecrnorm(lbfgsfloat_t* s, const lbfgsfloat_t *x, const int n)
{
	vecnorm(s, x, n);
	*s = (lbfgsfloat_t)(1.0 / *s);
}

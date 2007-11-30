/*
 *      C port of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
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

/* $Id$ */

#ifndef	__LBFGS_H__
#define	__LBFGS_H__

#ifdef	__cplusplus
extern "C" {
#endif/*__cplusplus*/

/*
 * The default precision of floating point values is 64bit (double).
 */
#ifndef	LBFGS_FLOAT
#define	LBFGS_FLOAT		64
#endif/*LBFGS_FLOAT*/

/*
 * Activate optimization routines for IEEE754 floating point values.
 */
#ifndef	LBFGS_IEEE_FLOAT
#define	LBFGS_IEEE_FLOAT	1
#endif/*LBFGS_IEEE_FLOAT*/

#if		LBFGS_FLOAT == 32
typedef float lbfgsfloatval_t;

#elif	LBFGS_FLOAT == 64
typedef double lbfgsfloatval_t;

#else
#error "liblbfgs supports single (float; LBFGS_FLOAT = 32) or double (double; LBFGS_FLOAT=64) precision only."

#endif


enum {
	LBFGSFALSE = 0,
	LBFGSTRUE,
	LBFGSERR_LOGICERROR = -1024,
	LBFGSERR_OUTOFMEMORY,
	LBFGSERR_CANCELED,
	LBFGSERR_INVALID_N,
	LBFGSERR_INVALID_N_SSE,
	LBFGSERR_INVALID_MINSTEP,
	LBFGSERR_INVALID_MAXSTEP,
	LBFGSERR_INVALID_FTOL,
	LBFGSERR_INVALID_GTOL,
	LBFGSERR_INVALID_XTOL,
	LBFGSERR_INVALID_MAXLINESEARCH,
	LBFGSERR_INVALID_ORTHANTWISE,
	LBFGSERR_OUTOFINTERVAL,
	LBFGSERR_INCORRECT_TMINMAX,
	LBFGSERR_ROUNDING_ERROR,
	LBFGSERR_MINIMUMSTEP,
	LBFGSERR_MAXIMUMSTEP,
	LBFGSERR_MAXIMUMITERATION,
	LBFGSERR_WIDTHTOOSMALL,
	LBFGSERR_INVALIDPARAMETERS,
	LBFGSERR_INCREASEGRADIENT,
};

typedef lbfgsfloatval_t (*lbfgs_evaluate_t)(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	);

typedef int (*lbfgs_progress_t)(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
	);

struct tag_lbfgs_parameter {
	/** The size of corrections. */
	int				m;

	/** Epsilon. */
	lbfgsfloatval_t	epsilon;

	int				max_linesearch;

	lbfgsfloatval_t	min_step;
	lbfgsfloatval_t	max_step;
	lbfgsfloatval_t	ftol;
	lbfgsfloatval_t	gtol;
	lbfgsfloatval_t	xtol;

	lbfgsfloatval_t	orthantwise_c;
};
typedef struct tag_lbfgs_parameter lbfgs_parameter_t;

/*
A user must implement a function compatible with ::lbfgs_evaluate_t (evaluation
callback) and pass the pointer to the callback function to lbfgs() arguments.
Similarly, a user can implement a function compatible with ::lbfgs_progress_t
(progress callback) to obtain the current progress (e.g., variables, function
value, ||G||, etc) and to cancel the iteration process if necessary.
Implementation of a progress callback is optional: a user can pass \c NULL if
progress notification is not necessary.

In addition, a user must preserve two requirements:
	- The number of variables must be multiples of 16 (this is not 4).
	- The memory block of variable array ::x must be aligned to 16.

This algorithm terminates an optimization
when:

	||G|| < \epsilon \cdot \max(1, ||x||) .

In this formula, ||.|| denotes the Euclidean norm.
*/
int lbfgs(
	const int n,
	lbfgsfloatval_t *x,
	lbfgs_evaluate_t proc_evaluate,
	lbfgs_progress_t proc_progress,
	void *instance,
	lbfgs_parameter_t *param
	);

int lbfgs_ow(
	const int n,
	lbfgsfloatval_t *x,
	lbfgs_evaluate_t proc_evaluate,
	lbfgs_progress_t proc_progress,
	void *instance,
	lbfgs_parameter_t *param
	);

void lbfgs_parameter_init(lbfgs_parameter_t *param);

#ifdef	__cplusplus
}
#endif/*__cplusplus*/



/**
@mainpage C port of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)

@section intro Introduction

This library is a C port of the implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
The original FORTRAN source code is available at:
http://www.ece.northwestern.edu/~nocedal/lbfgs.html

The L-BFGS method solves the unconstrainted minimization problem,

<pre>
    minimize F(x), x = (x1, x2, ..., xN),
</pre>

only if the object function F(x) and its gradient G(x) are computable. The
Newton's method, which is a well-known algorithm for the optimization,
requires computation or approximation of the inverse of the hessian matrix of
the object function in order to find the point where the gradient G(X) = 0.
The computational cost for the inverse hessian matrix is expensive especially
when the object function depends on a large number of variables. The L-BFGS
method approximates the inverse hessian matrix efficiently by using
information from last m iterations. This innovation saves the memory storage
and computational time a lot for large-scaled problems.

Among the various ports of L-BFGS, this library provides several features:
- <b>Clean C code</b>:
  Unlike C codes generated automatically by f2c (Fortran 77 into C converter),
  this port includes changes based on my interpretations, improvements,
  optimizations, and clean-ups so that the ported code would be well-suited
  for a C code. In addition to comments inherited from the original code,
  a number of comments were added through my interpretations.
- <b>Callback interface</b>:
  The library receives function and gradient values via a callback interface.
  The library also notifies the progress of the optimization by invoking a
  callback function. In the original implementation, a user had to set
  function and gradient values every time the function returns for obtaining
  updated values.
- <b>Thread safe</b>:
  The library is thread-safe, which is the secondary gain from the callback
  interface.
- <b>Configurable precision</b>: A user can choose single-precision (float)
  or double-precision (double) accuracy by changing ::LBFGS_FLOAT macro.
- <b>SSE/SSE2 optimization</b>:
  This library includes SSE/SSE2 optimization (written in compiler intrinsics)
  for vector arithmetic operations on Intel/AMD processors. The library uses
  SSE for float values and SSE2 for double values. The SSE/SSE2 optimization
  routine is disabled by default; compile the library with __SSE__ symbol
  defined to activate the optimization routine.

@section sample Sample code

@code

#include <stdio.h>
#include <lbfgs.h>

static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	)
{
	int i;
	lbfgsfloatval_t fx = 0.0;

	for (i = 0;i < n;i += 2) {
		lbfgsfloatval_t t1 = 1.0 - x[i];
		lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
		if (g != NULL) {
			g[i+1] = 20.0 * t2;
			g[i] = -2.0 * (x[i] * g[i+1] + t1);
		}
		fx += t1 * t1 + t2 * t2;
	}
	return fx;
}

static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
	)
{
	printf("Iteration %d:\n", k);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}

#define	N	4096

int main(int argc, char *argv)
{
	int i, ret = 0;
	lbfgsfloatval_t x[N];

	// Initialize the variables.
	for (i = 0;i < N;i += 2) {
		x[i] = -1.2;
		x[i+1] = 1.0;
	}

	// Start the L-BFGS optimization.
	// This will invoke the callback functions evaluate() and progress().
	ret = lbfgs(N, x, evaluate, progress, NULL, NULL);

	// Report the result.
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n",
		evaluate(NULL, x, NULL, N, 0), x[0], x[1]);

	return 0;
}

@endcode

@section download Download

- <a href="http://www.chokkan.org/software/dist/liblbfgs-1.0.zip">Source code</a>

libLBFGS is distributed under the term of the
<a href="http://opensource.org/licenses/mit-license.php">MIT license</a>.

@section ack Acknowledgements

The L-BFGS algorithm is described in:
	- Jorge Nocedal.
	  Updating Quasi-Newton Matrices with Limited Storage.
	  <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
	- Dong C. Liu and Jorge Nocedal.
	  On the limited memory BFGS method for large scale optimization.
	  <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

The line search algorithm used in this implementation is described in:
	- Jorge J. More and David J. Thuente.
	  Line search algorithm with guaranteed sufficient decrease.
	  <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
	  pp. 286-307, 1994.

Finally I would like to thank the original author, Jorge Nocedal, who has been
distributing the effieicnt and explanatory implementation in an open source
licence.

@section reference Reference

- <a href="http://www.ece.northwestern.edu/~nocedal/lbfgs.html">L-BFGS</a> by Jorge Nocedal.
- <a href="http://chasen.org/~taku/software/misc/lbfgs/">C port (via f2c)</a> by Taku Kudo.
- <a href="http://www.alglib.net/optimization/lbfgs.php">C#/C++/Delphi/VisualBasic6 port</a> in ALGLIB.
- <a href="http://cctbx.sourceforge.net/">Computational Crystallography Toolbox</a> includes
  <a href="http://cctbx.sourceforge.net/current_cvs/c_plus_plus/namespacescitbx_1_1lbfgs.html">scitbx::lbfgs</a>.
*/

#endif/*__LBFGS_H__*/

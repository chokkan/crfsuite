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

/* $Id:$ */

/*
This library is a C port of the FORTRAN implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
The original FORTRAN source code is available at:
http://www.ece.northwestern.edu/~nocedal/lbfgs.html

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

I would like to thank the original author, Jorge Nocedal, who has been
distributing the effieicnt and explanatory implementation in an open source
licence.
*/

#ifdef	HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <lbfgs.h>

#ifdef	_MSC_VER
#define	inline	__inline
typedef unsigned int uint32_t;
#endif/*_MSC_VER*/

#if		defined(__SSE__) && LBFGS_FLOAT == 32
/* Use SSE optimization for 32bit float precision. */
#include "arithmetic_sse_float.h"

#elif	defined(__SSE__) && LBFGS_FLOAT == 64
/* Use SSE2 optimization for 64bit double precision. */
#include "arithmetic_sse_double.h"

#else
/* No CPU specific optimization. */
#include "arithmetic_ansi.h"

#endif

#define min2(a, b)		((a) <= (b) ? (a) : (b))
#define max2(a, b)		((a) >= (b) ? (a) : (b))
#define	max3(a, b, c)	max2(max2((a), (b)), (c));

struct tag_iteration_data {
	lbfgsfloat_t rho;
	lbfgsfloat_t alpha;
	lbfgsfloat_t *s;		/* [n] */
	lbfgsfloat_t *y;		/* [n] */
};
typedef struct tag_iteration_data iteration_data_t;

static const lbfgs_parameter_t _defparam = {
	5, 1e-5, 20,
	1e-20, 1e20, 1e-4, 0.9, 1.0e-16,
};

/* Forward function declarations. */

static int line_search(
	int n,
	lbfgsfloat_t *x,
	lbfgsfloat_t *f,
	lbfgsfloat_t *g,
	lbfgsfloat_t *s,
	lbfgsfloat_t *stp,
	lbfgsfloat_t *wa,
	lbfgs_evaluate_t proc_evaluate,
	void *instance,
	const lbfgs_parameter_t *param
	);

static int update_trial_interval(
	lbfgsfloat_t *x,
	lbfgsfloat_t *fx,
	lbfgsfloat_t *dx,
	lbfgsfloat_t *y,
	lbfgsfloat_t *fy,
	lbfgsfloat_t *dy,
	lbfgsfloat_t *t,
	lbfgsfloat_t *ft,
	lbfgsfloat_t *dt,
	const lbfgsfloat_t tmin,
	const lbfgsfloat_t tmax,
	int *brackt
	);





int lbfgs(
	const int n,
	lbfgsfloat_t *x,
	lbfgs_evaluate_t proc_evaluate,
	lbfgs_progress_t proc_progress,
	void *instance,
	lbfgs_parameter_t *_param
	)
{
	int ret;
	int i, j, k, ls, end, bound;
	lbfgsfloat_t step;

	/* Constant parameters and their default values. */
	const lbfgs_parameter_t* param = (_param != NULL) ? _param : &_defparam;
	const int m = param->m;

	lbfgsfloat_t *g = NULL, *h = NULL, *w = NULL;
	iteration_data_t *lm = NULL, *it = NULL;
	lbfgsfloat_t ys, yy;
	lbfgsfloat_t xnorm, gnorm, beta;
	lbfgsfloat_t fx;

	/* Check the input parameters for errors. */
	if (n <= 0) {
		return LBFGSERR_INVALID_N;
	}
#if		defined(__SSE__)
	if (n % 8 != 0) {
		return LBFGSERR_INVALID_N_SSE;
	}
#endif/*defined(__SSE__)*/
	if (param->min_step < 0.) {
		return LBFGSERR_INVALID_MINSTEP;
	}
	if (param->max_step < param->min_step) {
		return LBFGSERR_INVALID_MAXSTEP;
	}
	if (param->ftol < 0.) {
		return LBFGSERR_INVALID_FTOL;
	}
	if (param->gtol < 0.) {
		return LBFGSERR_INVALID_GTOL;
	}
	if (param->xtol < 0.) {
		return LBFGSERR_INVALID_XTOL;
	}
	if (param->max_linesearch <= 0) {
		return LBFGSERR_INVALID_MAXLINESEARCH;
	}

	/* Allocate working space. */
	g = (lbfgsfloat_t*)vecalloc(n * sizeof(lbfgsfloat_t));
	h = (lbfgsfloat_t*)vecalloc(n * sizeof(lbfgsfloat_t));
	w = (lbfgsfloat_t*)vecalloc(n * sizeof(lbfgsfloat_t));
	if (g == NULL || h == NULL || w == NULL) {
		ret = LBFGSERR_OUTOFMEMORY;
		goto lbfgs_exit;
	}

	/* Allocate limited memory storage. */
	lm = (iteration_data_t*)vecalloc(m * sizeof(iteration_data_t));
	if (lm == NULL) {
		ret = LBFGSERR_OUTOFMEMORY;
		goto lbfgs_exit;
	}

	/* Initialize the limited memory. */
	for (i = 0;i < m;++i) {
		it = &lm[i];
		it->alpha = 0;
		it->rho = 0;
		it->s = (lbfgsfloat_t*)vecalloc(n * sizeof(lbfgsfloat_t));
		it->y = (lbfgsfloat_t*)vecalloc(n * sizeof(lbfgsfloat_t));
		if (it->s == NULL || it->y == NULL) {
			ret = LBFGSERR_OUTOFMEMORY;
			goto lbfgs_exit;
		}
	}

	/* Evaluate the function value and its gradient. */
    fx = proc_evaluate(instance, x, g, n, 0);

	/* Initialize the hessian matrix H_0 as the identity matrix. */
	vecset(h, 1.0, n);

	for (i = 0;i < n;++i) {
		lm[0].s[i] = -g[i] * h[i];
	}

	/* step = 1.0 / sqrt(vecdot(g, g, n)) */
	vecrnorm(&step, g, n);

	k = 1;
	end = 0;
	for (;;) {
		/* Store the current gradient vector to the work area. */
		veccpy(w, g, n);

		/* Current limited memory. */
		it = &lm[end];

		/* Search for the optimal step. */
		ls = line_search(
			n, x, &fx, g, it->s, &step, h, proc_evaluate, instance, param);
		if (ls < 0) {
			ret = ls;
			goto lbfgs_exit;
		}

		/* Compute x and g norms. */
		vecnorm(&gnorm, g, n);
		vecnorm(&xnorm, x, n);

		/* Report the progress. */
		if (proc_progress) {
			if (proc_progress(instance, x, g, fx, xnorm, gnorm, step, n, k, ls) != 0) {
				ret = LBFGSERR_CANCELED;
				goto lbfgs_exit;
			}
		}

		/*
			Convergence test.
			The criterion is given by the following formula:
				|g(x)| / \max(1, |x|) < \epsilon
		 */
		if (xnorm < 1.0) xnorm = 1.0;
		if (gnorm / xnorm <= param->epsilon) {
			/* Convergence. */
			ret = 0;
			break;
		}

		/*
			Update vectors s and y:
				s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
				y_{k+1} = g_{k+1} - g_{k}.

			Before this point, the vector lm[end].s is equivalent to d_{k},
			and the work area w stores the previous gradient, g_{k}.
		 */
		vecscale(it->s, step, n);
		vecdiff(it->y, g, w, n);

		/*
			Update rho:
				\rho = 1/(y^t \cdot s).
		 */
		vecdot(&ys, it->y, it->s, n);
		it->rho = 1.0 / ys;

		/*
			Scale the hessian matrix H_0 with Cholesky factor.
		 */
		vecdot(&yy, it->y, it->y, n);
		vecset(h, ys / yy, n);

		/*
			Recursive formula to compute H \cdot g described in page 779 of:
				Jorge Nocedal.
				Updating Quasi-Newton Matrices with Limited Storage.
				Mathematics of Computation, Vol. 35, No. 151,
				pp. 773--782, 1980.
		 */
		bound = (m <= k) ? m : k;
		++k;
		end = (end + 1) % m;

		vecncpy(w, g, n);

		j = end;
		for (i = 0;i < bound;++i) {
			j = (j + m - 1) % m;	/* if (--j == -1) j = m-1; */
			it = &lm[j];
			/* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
			vecdot(&it->alpha, it->s, w, n);
			it->alpha *= it->rho;
			/* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
			vecadd(w, it->y, -it->alpha, n);
		}

		/* \gamma_0 = H \cdot q_{0}. */
		vecmul(w, h, n);

		for (i = 0;i < bound;++i) {
			it = &lm[j];
			/* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
			vecdot(&beta, it->y, w, n);
			beta *= it->rho;
			/* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
			vecadd(w, it->s, it->alpha - beta, n);
			j = (j + 1) % m;		/* if (++j == m) j = 0; */
		}

		/*
			Store d_{k} = -H_{k} \cdot g_{k} into lm[end].s.
		 */
		it = &lm[end];
		veccpy(it->s, w, n);

		/*
			We try step = 1 first.
		 */
		step = 1.0;
	}

lbfgs_exit:
	/* Free memory blocks used by this function. */
	if (lm != NULL) {
		for (i = 0;i < m;++i) {
			vecfree(lm[i].s);
			vecfree(lm[i].y);
		}
		vecfree(lm);
	}
	vecfree(w);
	vecfree(h);
	vecfree(g);

	return ret;
}



static int line_search(
	int n,
	lbfgsfloat_t *x,
	lbfgsfloat_t *f,
	lbfgsfloat_t *g,
	lbfgsfloat_t *s,
	lbfgsfloat_t *stp,
	lbfgsfloat_t *wa,
	lbfgs_evaluate_t proc_evaluate,
	void *instance,
	const lbfgs_parameter_t *param
	)
{
	int count = 0;
	int brackt, stage1, uinfo = 0;
	lbfgsfloat_t dg;
	lbfgsfloat_t stx, fx, dgx;
	lbfgsfloat_t sty, fy, dgy;
	lbfgsfloat_t fxm, dgxm, fym, dgym, fm, dgm;
	lbfgsfloat_t finit, ftest1, dginit, dgtest;
	lbfgsfloat_t width, prev_width;
	lbfgsfloat_t stmin, stmax;

	/* Check the input parameters for errors. */
	if (*stp <= 0.) {
		return LBFGSERR_INVALIDPARAMETERS;
	}

	/*
		Compute the initial gradient in the search direction
		and check that s points to a descent direction.
	 */
	vecdot(&dginit, g, s, n);
	if (0 < dginit) {
		return LBFGSERR_INCREASEGRADIENT;
	}

	/* Initialize local variables. */
	brackt = 0;
	stage1 = 1;
	finit = *f;
	dgtest = param->ftol * dginit;
	width = param->max_step - param->min_step;
	prev_width = 2.0 * width;

	/* Copy the value of x to the work area. */
	veccpy(wa, x, n);

	/*
		The variables stx, fx, dgx contain the values of the step,
		function, and directional derivative at the best step.
		The variables sty, fy, dgy contain the value of the step,
		function, and derivative at the other endpoint of
		the interval of uncertainty.
		The variables stp, f, dg contain the values of the step,
		function, and derivative at the current step.
	*/
	stx = sty = 0.;
	fx = fy = finit;
	dgx = dgy = dginit;

	for (;;) {
		/*
			Set the minimum and maximum steps to correspond to the
			present interval of uncertainty.
		 */
		if (brackt) {
			stmin = min2(stx, sty);
			stmax = max2(stx, sty);
		} else {
			stmin = stx;
			stmax = *stp + 4.0 * (*stp - stx);
		}

		/* Clip the step in the range of [stpmin, stpmax]. */
		if (*stp < param->min_step) *stp = param->min_step;
		if (param->max_step < *stp) *stp = param->max_step;

		/*
			If an unusual termination is to occur then let
			stp be the lowest point obtained so far.
		 */
		if ((brackt && ((*stp <= stmin || stmax <= *stp) || param->max_linesearch <= count + 1 || uinfo != 0)) || (brackt && (stmax - stmin <= param->xtol * stmax))) {
			*stp = stx;
		}

		/*
			Compute the current value of x:
				x <- x + (*stp) * s.
		 */
		veccpy(x, wa, n);
		vecadd(x, s, *stp, n);

		/* Evaluate the function and gradient values. */
		*f = proc_evaluate(instance, x, g, n, *stp);

		++count;

		vecdot(&dg, g, s, n);
		ftest1 = finit + *stp * dgtest;

		/* Test for errors and convergence. */
		if (brackt && ((*stp <= stmin || stmax <= *stp) || uinfo != 0)) {
			/* Rounding errors prevent further progress. */
			return LBFGSERR_ROUNDING_ERROR;
		}
		if (*stp == param->max_step && *f <= ftest1 && dg <= dgtest) {
			/* The step is the maximum value. */
			return LBFGSERR_MAXIMUMSTEP;
		}
		if (*stp == param->min_step && (ftest1 < *f || dgtest <= dg)) {
			/* The step is the minimum value. */
			return LBFGSERR_MINIMUMSTEP;
		}
		if (brackt && (stmax - stmin) <= param->xtol * stmax) {
			/* Relative width of the interval of uncertainty is at most xtol. */
			return LBFGSERR_WIDTHTOOSMALL;
		}
		if (param->max_linesearch <= count) {
			/* Maximum number of iteration. */
			return LBFGSERR_MAXIMUMITERATION;
		}
		if (*f <= ftest1 && fabs(dg) <= param->gtol * (-dginit)) {
			/* The sufficient decrease condition and the directional derivative condition hold. */
			return count;
		}

		/*
			In the first stage we seek a step for which the modified
			function has a nonpositive value and nonnegative derivative.
		 */
		if (stage1 && *f <= ftest1 && min2(param->ftol, param->gtol) * dginit <= dg) {
			stage1 = 0;
		}

		/*
			A modified function is used to predict the step only if
			we have not obtained a step for which the modified
			function has a nonpositive function value and nonnegative
			derivative, and if a lower function value has been
			obtained but the decrease is not sufficient.
		 */
		if (stage1 && ftest1 < *f && *f <= fx) {
		    /* Define the modified function and derivative values. */
			fm = *f - *stp * dgtest;
			fxm = fx - stx * dgtest;
			fym = fy - sty * dgtest;
			dgm = dg - dgtest;
			dgxm = dgx - dgtest;
			dgym = dgy - dgtest;

			/*
				Call update_trial_interval() to update the interval of
				uncertainty and to compute the new step.
			 */
			uinfo = update_trial_interval(
				&stx, &fxm, &dgxm,
				&sty, &fym, &dgym,
				stp, &fm, &dgm,
				stmin, stmax, &brackt
				);

			/* Reset the function and gradient values for f. */
			fx = fxm + stx * dgtest;
			fy = fym + sty * dgtest;
			dgx = dgxm + dgtest;
			dgy = dgym + dgtest;
		} else {
			/*
				Call update_trial_interval() to update the interval of
				uncertainty and to compute the new step.
			 */
			uinfo = update_trial_interval(
				&stx, &fx, &dgx,
				&sty, &fy, &dgy,
				stp, f, &dg,
				stmin, stmax, &brackt
				);
		}

		/*
			Force a sufficient decrease in the interval of uncertainty.
		 */
		if (brackt) {
			if (0.66 * prev_width <= fabs(sty - stx)) {
				*stp = stx + 0.5 * (sty - stx);
			}
			prev_width = width;
			width = fabs(sty - stx);
		}
	}

	return LBFGSERR_LOGICERROR;
}



/**
 * Define the local variables for computing minimizers.
 */
#define	USES_MINIMIZER \
	lbfgsfloat_t a, d, gamma, theta, p, q, r, s;

/**
 * Find a minimizer of an interpolated cubic function.
 *	@param	cm		The minimizer of the interpolated cubic.
 *	@param	u		The value of one point, u.
 *	@param	fu		The value of f(u).
 *	@param	du		The value of f'(u).
 *	@param	v		The value of another point, v.
 *	@param	fv		The value of f(v).
 *	@param	du		The value of f'(v).
 */
#define	CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv) \
	d = (v) - (u); \
	theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
	p = fabs(theta); \
	q = fabs(du); \
	r = fabs(dv); \
	s = max3(p, q, r); \
	/* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
	a = theta / s; \
	gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s)); \
	if ((v) < (u)) gamma = -gamma; \
	p = gamma - (du) + theta; \
	q = gamma - (du) + gamma + (dv); \
	r = p / q; \
	(cm) = (u) + r * d;

/**
 * Find a minimizer of an interpolated cubic function.
 *	@param	cm		The minimizer of the interpolated cubic.
 *	@param	u		The value of one point, u.
 *	@param	fu		The value of f(u).
 *	@param	du		The value of f'(u).
 *	@param	v		The value of another point, v.
 *	@param	fv		The value of f(v).
 *	@param	du		The value of f'(v).
 *	@param	xmin	The maximum value.
 *	@param	xmin	The minimum value.
 */
#define	CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax) \
	d = (v) - (u); \
	theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
	p = fabs(theta); \
	q = fabs(du); \
	r = fabs(dv); \
	s = max3(p, q, r); \
	/* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
	a = theta / s; \
	gamma = s * sqrt(max2(0, a * a - ((du) / s) * ((dv) / s))); \
	if ((u) < (v)) gamma = -gamma; \
	p = gamma - (dv) + theta; \
	q = gamma - (dv) + gamma + (du); \
	r = p / q; \
	if (r < 0. && gamma != 0.) { \
		(cm) = (v) - r * d; \
	} else if (a < 0) { \
		(cm) = (xmax); \
	} else { \
		(cm) = (xmin); \
	}

/**
 * Find a minimizer of an interpolated quadratic function.
 *	@param	qm		The minimizer of the interpolated quadratic.
 *	@param	u		The value of one point, u.
 *	@param	fu		The value of f(u).
 *	@param	du		The value of f'(u).
 *	@param	v		The value of another point, v.
 *	@param	fv		The value of f(v).
 */
#define	QUARD_MINIMIZER(qm, u, fu, du, v, fv) \
	a = (v) - (u); \
	(qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

/**
 * Find a minimizer of an interpolated quadratic function.
 *	@param	qm		The minimizer of the interpolated quadratic.
 *	@param	u		The value of one point, u.
 *	@param	du		The value of f'(u).
 *	@param	v		The value of another point, v.
 *	@param	dv		The value of f'(v).
 */
#define	QUARD_MINIMIZER2(qm, u, du, v, dv) \
	a = (u) - (v); \
	(qm) = (v) + (dv) / ((dv) - (du)) * a;

/**
 * Update a safeguarded trial value and interval for line search.
 *
 *	The parameter x represents the step with the least function value.
 *	The parameter t represents the current step. This function assumes
 *	that the derivative at the point of x in the direction of the step.
 *	If the bracket is set to true, the minimizer has been bracketed in
 *	an interval of uncertainty with endpoints between x and y.
 *
 *	@param	x		The pointer to the value of one endpoint.
 *	@param	fx		The pointer to the value of f(x).
 *	@param	dx		The pointer to the value of f'(x).
 *	@param	y		The pointer to the value of another endpoint.
 *	@param	fy		The pointer to the value of f(y).
 *	@param	dy		The pointer to the value of f'(y).
 *	@param	t		The pointer to the value of the trial value, t.
 *	@param	ft		The pointer to the value of f(t).
 *	@param	dt		The pointer to the value of f'(t).
 *	@param	tmin	The minimum value for the trial value, t.
 *	@param	tmax	The maximum value for the trial value, t.
 *	@param	brackt	The pointer to the predicate if the trial value is
 *					bracketed.
 *	@retval	int		Status value. Zero indicates a normal termination.
 *	
 *	@see
 *		Jorge J. More and David J. Thuente. Line search algorithm with
 *		guaranteed sufficient decrease. ACM Transactions on Mathematical
 *		Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
 */
static int update_trial_interval(
	lbfgsfloat_t *x,
	lbfgsfloat_t *fx,
	lbfgsfloat_t *dx,
	lbfgsfloat_t *y,
	lbfgsfloat_t *fy,
	lbfgsfloat_t *dy,
	lbfgsfloat_t *t,
	lbfgsfloat_t *ft,
	lbfgsfloat_t *dt,
	const lbfgsfloat_t tmin,
	const lbfgsfloat_t tmax,
	int *brackt
	)
{
	int bound;
	int dsign = fsigndiff(dt, dx);
	lbfgsfloat_t mc;	/* minimizer of an interpolated cubic. */
	lbfgsfloat_t mq;	/* minimizer of an interpolated quadratic. */
	lbfgsfloat_t newt;	/* new trial value. */
	USES_MINIMIZER;		/* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

	/* Check the input parameters for errors. */
	if (*brackt) {
		if (*t <= min2(*x, *y) || max2(*x, *y) <= *t) {
			/* The trival value t is out of the interval. */
			return LBFGSERR_OUTOFINTERVAL;
		}
		if (0. <= *dx * (*t - *x)) {
			/* The function must decrease from x. */
			return LBFGSERR_INCREASEGRADIENT;
		}
		if (tmax < tmin) {
			/* Incorrect tmin and tmax specified. */
			return LBFGSERR_INCORRECT_TMINMAX;
		}
	}

	/*
		Trial value selection.
	 */
	if (*fx < *ft) {
		/*
			Case 1: a higher function value.
			The minimum is brackt. If the cubic minimizer is closer
			to x than the quadratic one, the cubic one is taken, else
			the average of the minimizers is taken.
		 */
		*brackt = 1;
		bound = 1;
		CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
		QUARD_MINIMIZER(mq, *x, *fx, *dx, *t, *ft);
		if (fabs(mc - *x) < fabs(mq - *x)) {
			newt = mc;
		} else {
			newt = mc + 0.5 * (mq - mc);
		}
	} else if (dsign) {
		/*
			Case 2: a lower function value and derivatives of
			opposite sign. The minimum is brackt. If the cubic
			minimizer is closer to x than the quadratic (secant) one,
			the cubic one is taken, else the quadratic one is taken.
		 */
		*brackt = 1;
		bound = 0;
		CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
		QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
		if (fabs(mc - *t) > fabs(mq - *t)) {
			newt = mc;
		} else {
			newt = mq;
		}
	} else if (fabs(*dt) < fabs(*dx)) {
		/*
			Case 3: a lower function value, derivatives of the
			same sign, and the magnitude of the derivative decreases.
			The cubic minimizer is only used if the cubic tends to
			infinity in the direction of the minimizer or if the minimum
			of the cubic is beyond t. Otherwise the cubic minimizer is
			defined to be either tmin or tmax. The quadratic (secant)
			minimizer is also computed and if the minimum is brackt
			then the the minimizer closest to x is taken, else the one
			farthest away is taken.
		 */
		bound = 1;
		CUBIC_MINIMIZER2(mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
		QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
		if (*brackt) {
			if (fabs(*t - mc) < fabs(*t - mq)) {
				newt = mc;
			} else {
				newt = mq;
			}
		} else {
			if (fabs(*t - mc) > fabs(*t - mq)) {
				newt = mc;
			} else {
				newt = mq;
			}
		}
	} else {
		/*
			Case 4: a lower function value, derivatives of the
			same sign, and the magnitude of the derivative does
			not decrease. If the minimum is not brackt, the step
			is either tmin or tmax, else the cubic minimizer is taken.
		 */
		bound = 0;
		if (*brackt) {
			CUBIC_MINIMIZER(newt, *t, *ft, *dt, *y, *fy, *dy);
		} else if (*x < *t) {
			newt = tmax;
		} else {
			newt = tmin;
		}
	}

	/*
		Update the interval of uncertainty. This update does not
		depend on the new step or the case analysis above.

		- Case a: if f(x) < f(t),
			x <- x, y <- t.
		- Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
			x <- t, y <- y.
		- Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0, 
			x <- t, y <- x.
	 */
	if (*fx < *ft) {
		/* Case a */
		*y = *t;
		*fy = *ft;
		*dy = *dt;
	} else {
		/* Case c */
		if (dsign) {
			*y = *x;
			*fy = *fx;
			*dy = *dx;
		}
		/* Cases b and c */
		*x = *t;
		*fx = *ft;
		*dx = *dt;
	}

	/* Clip the new trial value in [tmin, tmax]. */
	if (tmax < newt) newt = tmax;
	if (newt < tmin) newt = tmin;

	/*
		Redefine the new trial value if it is close to the upper bound
		of the interval.
	 */
	if (*brackt && bound) {
		mq = *x + 0.66 * (*y - *x);
		if (*x < *y) {
			if (mq < newt) newt = mq;
		} else {
			if (newt < mq) newt = mq;
		}
	}

	/* Return the new trial value. */
	*t = newt;
	return 0;
}

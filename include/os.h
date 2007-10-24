/*
 *		Compatibility stuff among operating systems and compilers.
 *
 *	Copyright (C) 2007 Naoaki Okazaki
 *
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* $Id:$ */

#ifndef	__OS_H__
#define	__OS_H__


typedef double float_t;
#define	FLOAT_MAX	DBL_MAX


//#define	__SSE__ 1
#define	LBFGS_FLOAT		64
typedef double float_t;

#ifdef	_MSC_VER
/* Microsoft Visual C/C++ specific */

#define	_CRT_SECURE_NO_WARNINGS 1
#pragma warning(disable : 4996)

#define	alloca	_alloca
#define	strdup	_strdup
#define	open	_open

#ifndef	__cplusplus
/* Microsoft Visual C specific */

#define	inline	__inline

#endif/*__cplusplus*/

#endif/*_MSC_VER*/

#endif/*__OS_H__*/

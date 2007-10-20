#ifndef	__OS_H__
#define	__OS_H__


typedef double float_t;
#define	FLOAT_MAX	DBL_MAX

#define	alloca	_alloca
#define	strdup	_strdup
#define	open	_open

#define	_CRT_SECURE_NO_WARNINGS 1

//#define	__SSE__ 1
#define	LBFGS_FLOAT		64
typedef double float_t;

#ifdef	_MSC_VER
/* Microsoft Visual C/C++ specific */

typedef unsigned int uint32_t;

#ifndef	__cplusplus
/* Microsoft Visual C specific */

#define	inline	__inline

#endif/*__cplusplus*/

#endif/*_MSC_VER*/

#endif/*__OS_H__*/

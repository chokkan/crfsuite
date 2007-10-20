/*
 *		A parser for command-line options.
 *
 *		Copyright (c) 2007 Naoaki Okazaki
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions (known as zlib license):
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Naoaki Okazaki <okazaki at chokkan dot org>
 *
 */

#ifndef	__OPTION_H__
#define	__OPTION_H__

#ifdef	__cplusplus
extern "C" {
#endif/*__cplusplus*/

typedef int (*option_handler_t)(void *instance, char c, const char *longname, const char *arg);

int option_parse(char * const argv[], int num_argv, option_handler_t handler, void *instance);
int option_strcmp(const char *option, const char *longname);

/** The begin of inline option map. */
#define	BEGIN_OPTION_MAP(name, type) \
	int name(void *instance, char __c, const char *__longname, const char *arg) \
	{ \
		int used_args = 0; \
		type *opt = (type *)instance; \
		if (0) { \

/** An entry of option map */
#define	ON_OPTION(test) \
			return used_args; \
		} else if (test) { \
			used_args = 0; \

#define	ON_OPTION_WITH_ARG(test) \
			return used_args; \
		} else if (test) { \
			used_args = 1; \

/** The end of option map implementation */
#define	END_OPTION_MAP() \
			return used_args; \
		} \
		return -1; \
	} \

/** A predicator for short options */
#define	SHORTOPT(x)		(__c == x)
/** A predicator for long options */
#define	LONGOPT(x)		(!__c && option_strcmp(__longname, x) == 0)

#ifdef	__cplusplus
}
#endif/*__cplusplus*/

#endif/*__OPTION_H__*/


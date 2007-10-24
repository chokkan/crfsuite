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

/* $Id:$ */

#include <os.h>
#include <stdlib.h>
#include <string.h>
#include "option.h"

int option_parse(char * const argv[], int num_argv, option_handler_t handler, void *instance)
{
	int i;

	for (i = 0;i < num_argv;++i) {
		const char *token = argv[i];
		if (*token++ == '-') {
			int ret = 0;
			const char *next_token = (i+1 < num_argv) ? argv[i+1] : "";
			if (!*token) {
				break;	/* Only '-' was found. */
			} else if (*token == '-') {
				const char *arg = strchr(++token, '=');
				if (arg) {
					arg++;
				} else {
					arg = next_token;
				}

				ret = handler(instance, 0, token, arg);
				if (ret < 0) {
					return -1;
				}
				if (arg == next_token) {
					i += ret;
				}
			} else {
				char c;
				while ((c = *token++) != '\0') {
					const char *arg = *token ? token : next_token;
					ret = handler(instance, c, token, arg);
					if (ret < 0) {
						return -1;
					}
					if (ret > 0) {
						if (arg == token) {
							token = "";
						} else {
							i++;
						}
					}
				} /* while */
			} /* else (*token == '-') */
		} else {
			break;	/* a non-option argument was fonud. */
		} 
	} /* for (i) */

	return i;
}

int option_strcmp(const char *option, const char *longname)
{
	const char *p = strchr(option, '=');
	return p ? strncmp(option, longname, p-option) : strcmp(option, longname);
}

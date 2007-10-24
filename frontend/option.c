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

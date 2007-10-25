/*
 *		Dump command for libCRF frontend.
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

/* $Id$ */

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "option.h"

#define	SAFE_RELEASE(obj)	if ((obj) != NULL) { (obj)->release(obj); (obj) = NULL; }

typedef struct {
	int help;
} dump_option_t;

static void dump_option_init(dump_option_t* opt)
{
	memset(opt, 0, sizeof(*opt));
}

static void dump_option_finish(dump_option_t* opt)
{
}

BEGIN_OPTION_MAP(parse_dump_options, dump_option_t)

	ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
		opt->help = 1;

END_OPTION_MAP()

static void show_usage(FILE *fp, const char *argv0, const char *command)
{
	fprintf(fp, "USAGE: %s %s [OPTIONS] <MODEL>\n", argv0, command);
	fprintf(fp, "Output the model stored in the file (MODEL) in a plain-text format\n");
	fprintf(fp, "\n");
	fprintf(fp, "OPTIONS:\n");
	fprintf(fp, "    -h, --help      Show the usage of this command and exit\n");
}

int main_dump(int argc, char *argv[], const char *argv0)
{
	int ret = 0, arg_used = 0;
	dump_option_t opt;
	const char *command = argv[0];
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	crf_model_t *model = NULL;

	/* Parse the command-line option. */
	dump_option_init(&opt);
	arg_used = option_parse(++argv, --argc, parse_dump_options, &opt);
	if (arg_used < 0) {
		ret = 1;
		goto force_exit;
	}

	/* Show the help message for this command if specified. */
	if (opt.help) {
		show_usage(fpo, argv0, command);
		goto force_exit;
	}

	/* Check for the existence of the model file. */
	if (argc <= arg_used) {
		fprintf(fpe, "ERROR: No model specified.\n");
		ret = 1;
		goto force_exit;
	}

	/* Create a model instance corresponding to the model file. */
	if (ret = crf_create_instance_from_file(argv[arg_used], &model)) {
		goto force_exit;
	}
		
	/* Dump the model. */
	if (ret = model->dump(model, fpo)) {
		goto force_exit;
	}

force_exit:
	SAFE_RELEASE(model);
	dump_option_finish(&opt);
	return ret;
}

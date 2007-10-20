#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "option.h"
#include "readdata.h"

typedef struct {
	char *input;
	char *model;
	int evaluate;

	int num_params;
	char **params;
} tagger_option_t;

static void tagger_option_init(tagger_option_t* opt)
{
	memset(opt, 0, sizeof(*opt));
}

static void tagger_option_finish(tagger_option_t* opt)
{
	int i;

	free(opt->model);

	for (i = 0;i < opt->num_params;++i) {
		free(opt->params[i]);
	}
	free(opt->params);
}

BEGIN_OPTION_MAP(parse_tagger_options, tagger_option_t)

	ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
		free(opt->model);
		opt->model = strdup(arg);

	ON_OPTION(SHORTOPT('e') || LONGOPT("evaluate"))
		opt->evaluate = 1;

	ON_OPTION_WITH_ARG(SHORTOPT('p') || LONGOPT("param"))
		opt->params = (char **)realloc(opt->params, sizeof(char*) * (opt->num_params + 1));
		opt->params[opt->num_params] = strdup(arg);
		++opt->num_params;

END_OPTION_MAP()

int tag(int argc, char *argv[])
{
	int ret = 0, arg_used = 0;
	tagger_option_t opt;
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	crf_tagger_t *tagger = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;
	int aid;

	/* Parse the command-line option. */
	tagger_option_init(&opt);
	arg_used = option_parse(argv, argc, parse_tagger_options, &opt);
	if (arg_used < 0) {
		return -1;
	}

	/* Set an input file. */
	if (arg_used < argc) {
		opt.input = strdup(argv[arg_used]);
	} else {
		opt.input = strdup("-");	/* STDIN. */
	}

	if (opt.model != NULL) {
		crf_create_tagger(opt.model, &tagger, &attrs, &labels);
	}

	aid = labels->to_id(labels, "I-NP");

force_exit:
	tagger_option_finish(&opt);
	return ret;
}

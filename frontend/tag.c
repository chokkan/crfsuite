#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "option.h"
#include "iwa.h"

#define	SAFE_RELEASE(obj)	if ((obj) != NULL) { (obj)->release(obj); (obj) = NULL; }

typedef struct {
	char *input;		/**< Filename of the input data. */
	char *model;		/**< Filename of the model. */
	int evaluate;		/**< Evaluate the tagging performance. */
	int quiet;
	int help;			/**< Show help message and exit. */

	int num_params;
	char **params;

	FILE *fpi;
	FILE *fpo;
	FILE *fpe;
} tagger_option_t;

static void tagger_option_init(tagger_option_t* opt)
{
	memset(opt, 0, sizeof(*opt));
	opt->fpi = stdin;
	opt->fpo = stdout;
	opt->fpe = stderr;
}

static void tagger_option_finish(tagger_option_t* opt)
{
	int i;

	free(opt->input);
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

	ON_OPTION(SHORTOPT('q') || LONGOPT("quiet"))
		opt->quiet = 1;

	ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
		opt->help = 1;

	ON_OPTION_WITH_ARG(SHORTOPT('p') || LONGOPT("param"))
		opt->params = (char **)realloc(opt->params, sizeof(char*) * (opt->num_params + 1));
		opt->params[opt->num_params] = strdup(arg);
		++opt->num_params;

END_OPTION_MAP()

static int output_result(FILE *fpo, crf_output_t *output, crf_dictionary_t *labels)
{
	int i;

	fprintf(fpo, "BOS\t%f\n", output->probability);
	for (i = 0;i < output->num_labels;++i) {
		char *label = NULL;
		labels->to_string(labels, output->labels[i], &label);
		fprintf(fpo, "%s\n", label);
		labels->free(labels, label);
	}
	fprintf(fpo, "EOS\n");
}

static int tag(tagger_option_t* opt, crf_model_t* model)
{
	int L = 0, ret = 0, lid = -1;
	crf_instance_t inst;
	crf_item_t item;
	crf_content_t cont;
	crf_output_t output;
	crf_evaluation_t eval;
	iwa_t* iwa = NULL;
	const iwa_token_t* token = NULL;
	crf_tagger_t *tagger = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;
	FILE *fp = NULL, *fpi = opt->fpi, *fpo = opt->fpo, *fpe = opt->fpe;

	/* Obtain the dictionary interface representing the labels in the model. */
	if (ret = model->get_labels(model, &labels)) {
		goto error_exit;
	}

	/* Obtain the dictionary interface representing the attributes in the model. */
	if (ret = model->get_attrs(model, &attrs)) {
		goto error_exit;
	}

	/* Obtain the tagger interface. */
	if (ret = model->get_tagger(model, &tagger)) {
		goto error_exit;
	}

	/* Initialize the objects for instance and evaluation. */
	L = labels->num(labels);
	crf_instance_init(&inst);
	crf_evaluation_init(&eval, L);

	/* Open the stream for the input data. */
	fp = (strcmp(opt->input, "-") == 0) ? fpi : fopen(opt->input, "r");
	if (fp == NULL) {
		fprintf(fpe, "ERROR: failed to open the stream for the input data,\n");
		fprintf(fpe, "  %s\n", opt->input);
		goto error_exit;
	}

	/* Open a IWA reader. */
	iwa = iwa_reader(fp);
	while (token = iwa_read(iwa), token != NULL) {
		switch (token->type) {
		case IWA_BOI:
			/* Initialize an item. */
			lid = -1;
			crf_item_init(&item);
			break;
		case IWA_EOI:
			/* Append the item to the instance. */
			crf_instance_append(&inst, &item, lid);
			crf_item_finish(&item);
			break;
		case IWA_ITEM:
			if (lid == -1) {
				/* The first field in a line presents a label. */
				lid = labels->to_id(labels, token->attr);
				if (lid < 0) lid = L;
			} else {
				/* Fields after the first field present attributes. */
				int aid = attrs->to_id(attrs, token->attr);
				/* Ignore attributes 'unknown' to the model. */
				if (0 <= aid) {
					/* Associate the attribute with the current item. */
					crf_content_set(&cont, aid, 1.0);
					crf_item_append_content(&item, &cont);
				}
			}
			break;
		case IWA_NONE:
			if (0 < inst.num_items) {
				/* Initialize the object to receive the tagging result. */
				crf_output_init(&output);

				/* Tag the instance. */
				if (ret = tagger->tag(tagger, &inst, &output)) {
					goto error_exit;
				}

				/* Accumulate the tagging performance. */
				if (opt->evaluate) {
					crf_evaluation_accmulate(&eval, &inst, &output);
				}

				if (!opt->quiet) {
					output_result(fpo, &output, labels);
				}

				crf_output_finish(&output);
				crf_instance_finish(&inst);
			}
			break;
		case IWA_COMMENT:
			break;
		}
	}

	/* Compute the performance if specified. */
	if (opt->evaluate) {
		crf_evaluation_compute(&eval);
		crf_evaluation_output(&eval, labels, fpo);
	}

error_exit:
	if (iwa != NULL) {
		iwa_delete(iwa);
		iwa = NULL;
	}

	if (fp != NULL && fp != fpi) {
		fclose(fp);
		fp = NULL;
	}

	crf_instance_finish(&inst);
	crf_evaluation_finish(&eval);

	SAFE_RELEASE(tagger);
	SAFE_RELEASE(attrs);
	SAFE_RELEASE(labels);

	return ret;
}

int main_tag(int argc, char *argv[], const char *argv0)
{
	int i;
	int ret = 0, arg_used = 0;
	tagger_option_t opt;
	const char *command = argv[0];
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	crf_model_t *model = NULL;

	/* Parse the command-line option. */
	tagger_option_init(&opt);
	arg_used = option_parse(++argv, --argc, parse_tagger_options, &opt);
	if (arg_used < 0) {
		return -1;
	}

	/* Set an input file. */
	if (arg_used < argc) {
		opt.input = strdup(argv[arg_used]);
	} else {
		opt.input = strdup("-");	/* STDIN. */
	}

	/* Read the model. */
	if (opt.model != NULL) {
		/* Create a model instance corresponding to the model file. */
		if (ret = crf_create_instance_from_file(opt.model, &model)) {
			goto force_exit;
		}
		
		tag(&opt, model);
		//model->dump(model, stdout);
	}

force_exit:
	SAFE_RELEASE(model);
	tagger_option_finish(&opt);
	return ret;
}

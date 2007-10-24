#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "option.h"
#include "iwa.h"

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

int main_tag(int argc, char *argv[])
{
	int i;
	int ret = 0, arg_used = 0;
	tagger_option_t opt;
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	crf_model_t *model = NULL;
	crf_tagger_t *tagger = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;

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

	/* Read the model. */
	if (opt.model != NULL) {
		/* Create a model instance corresponding to the model file. */
		if (ret = crf_create_instance_from_file(opt.model, &model)) {
			goto force_exit;
		}
		
		/* Obtain the dictionary interface for the labels in the model. */
		if (ret = model->get_labels(model, &labels)) {
			goto force_exit;
		}

		/* Obtain the dictionary interface for the attributes in the model. */
		if (ret = model->get_attrs(model, &attrs)) {
			goto force_exit;
		}

		/* Obtain the tagger interface. */
		if (ret = model->get_tagger(model, &tagger)) {
			goto force_exit;
		}

		//model->dump(model, stdout);
	}

	if (tagger != NULL && attrs != NULL && labels != NULL) {
		int lid = -1;
		crf_instance_t inst;
		crf_item_t item;
		crf_content_t cont;
		crf_output_t output;
		crf_evaluation_t eval;
		FILE *fp = NULL;
		iwa_t* iwa = NULL;
		const iwa_token_t* token = NULL;

		/* Initialize the instance.*/
		crf_instance_init(&inst);

		/* Initialize the evaluation table. */
		crf_evaluation_init(&eval);

		/* Open the input stream. */
		fp = (strcmp(opt.input, "-") == 0) ? fpi : fopen(opt.input, "r");
		if (fp == NULL) {

		}

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
					lid = labels->to_id(labels, token->attr);
				} else {
					int aid = attrs->to_id(attrs, token->attr);
					if (0 <= aid) {
						crf_content_init(&cont);
						cont.aid = aid;
						crf_item_append_content(&item, &cont);
					}
				}
				break;
			case IWA_NONE:
				/* Tag the instance. */
				if (0 < inst.num_items) {
					crf_output_init(&output);
					tagger->tag(tagger, &inst, &output);
					crf_evaluation_accmulate(&eval, &inst, &output);
					fprintf(fpo, "BOS (%f)\n", output.probability);
					for (i = 0;i < output.num_labels;++i) {
						char *label = NULL;
						labels->to_string(labels, output.labels[i], &label);
						fprintf(fpo, "%s\n", label);
						labels->free(labels, label);
					}
					fprintf(fpo, "EOS\n");
					crf_output_finish(&output);
					crf_instance_finish(&inst);
				}
				break;
			case IWA_COMMENT:
				break;
			}
		}

	}

force_exit:
	tagger_option_finish(&opt);
	return ret;
}

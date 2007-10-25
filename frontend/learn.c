/*
 *		Learn command for libCRF frontend.
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
#include <time.h>

#include <crf.h>
#include "option.h"
#include "readdata.h"

typedef struct {
	char *model;
	char *training;
	char *evaluation;
	char *output;
	int shuffle;
	int cross_validation;
	double holdout_validation;

	int num_params;
	char **params;
} learn_option_t;

static void learn_option_init(learn_option_t* opt)
{
	memset(opt, 0, sizeof(*opt));

	opt->shuffle = 0;
	opt->cross_validation = 0;
	opt->holdout_validation = 0;
	opt->num_params = 0;
}

static void learn_option_finish(learn_option_t* opt)
{
	int i;

	free(opt->model);
	free(opt->training);
	free(opt->evaluation);
	free(opt->output);

	for (i = 0;i < opt->num_params;++i) {
		free(opt->params[i]);
	}
	free(opt->params);
}

BEGIN_OPTION_MAP(parse_learn_options, learn_option_t)

	ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
		free(opt->model);
		opt->model = strdup(arg);

	ON_OPTION_WITH_ARG(SHORTOPT('o') || LONGOPT("output"))
		free(opt->output);
		opt->output = strdup(arg);

	ON_OPTION_WITH_ARG(SHORTOPT('t') || LONGOPT("test"))
		free(opt->evaluation);
		opt->evaluation = strdup(arg);

	ON_OPTION_WITH_ARG(SHORTOPT('x') || LONGOPT("cross-validation"))
		opt->cross_validation = atoi(arg);
		opt->holdout_validation = 0;

	ON_OPTION_WITH_ARG(SHORTOPT('h') || LONGOPT("holdout-validation"))
		opt->cross_validation = 0;
		opt->holdout_validation = atof(arg);

	ON_OPTION(SHORTOPT('s') || LONGOPT("shuffle"))
		opt->shuffle = 1;

	ON_OPTION_WITH_ARG(SHORTOPT('p') || LONGOPT("param"))
		opt->params = (char **)realloc(opt->params, sizeof(char*) * (opt->num_params + 1));
		opt->params[opt->num_params] = strdup(arg);
		++opt->num_params;

END_OPTION_MAP()


#define	MAX(a, b)	((a) < (b) ? (b) : (a))

static int trainer_callback(void *instance, const char *format, va_list args)
{
	vfprintf(stdout, format, args);
	return 0;
}

typedef struct {
	FILE *fpo;
	double best_accuracy;
	crf_data_t* data;
	crf_output_t out;
	crf_evaluation_t eval;
	crf_dictionary_t* attrs;
	crf_dictionary_t* labels;
} evaluation_data_t;

static int evaluate_callback(void *instance, crf_tagger_t* tagger)
{
	int i, best = 0;
	evaluation_data_t* ed = (evaluation_data_t*)instance;
	FILE *fpo = ed->fpo;
	crf_data_t* data = ed->data;
	crf_dictionary_t* labels = ed->labels;
	int total_correct = 0, total_model = 0, total_observation = 0;
	double accuracy = 0;
#if 0
	int j;
	int num_match = 0, num_extracted = 0, num_to_extract = 0;
	int lid_other = labels->to_id(labels, "O");
#endif

	/* Initialize the evaluation table. */
	crf_evaluation_clear(&ed->eval);

	/* Tag the evaluation data and accumulate the classification performance. */
	for (i = 0;i < data->num_instances;++i) {
		crf_instance_t* instance = &data->instances[i];

#if 1
		/* Tag an instance. */
		tagger->tag(tagger, instance, &ed->out);
		/* Compare the tagged output with the reference. */
		crf_evaluation_accmulate(&ed->eval, instance, &ed->out);
#endif

#if 0
		for (j = 0;j < instance->num_items;++j) {
			if (instance->labels[j] != lid_other) {
				break;
			}
		}
		if (j != instance->num_items) {
			++num_to_extract;
			for (j = 0;j < instance->num_items;++j) {
				if (ed->out.labels[j] != lid_other) {
					break;
				}
			}
			if (j != instance->num_items) {
				++num_extracted;
				for (j = 0;j < instance->num_items;++j) {
					if (ed->out.labels[j] != instance->labels[j]) {
						break;
					}
				}
				if (j == instance->num_items) {
					++num_match;
				}			
			}
		}
#endif
	}

#if 1
	/* Report the classification performance for each output label. */
	fprintf(fpo, "Performance (#match, #model, #ref), (prec, rec, f1):\n");
	for (i = 0;i < labels->num(labels);++i) {
		char *lstr = NULL;
		double precision = 0, recall = 0, f1 = 0;
		crf_label_evaluation_t* lev = &ed->eval.tbl[i];

		/* Sum the number of correct labels for accuracy calculation. */
		total_correct += lev->num_correct;
		total_model += lev->num_model;
		total_observation += lev->num_observation;

		/* Compute the precision, recall, f1-measure. */
		if (lev->num_model > 0) {
			precision = lev->num_correct * 100.0 / (double)lev->num_model;
		}
		if (lev->num_observation > 0) {
			recall = lev->num_correct * 100.0 / (double)lev->num_observation;
		}
		if (precision + recall > 0) {
			f1 = precision * recall * 2 / (precision + recall);
		}

		/* Output the performance for the label. */
		labels->to_string(labels, i, &lstr);
		fprintf(fpo, "    %s: (%d, %d, %d) (%3.2f, %3.2f, %3.2f)\n",
			lstr, lev->num_correct, lev->num_model, lev->num_observation, precision, recall, f1
			);
		labels->free(labels, lstr);
	}

	/* Compute the accuracy. */
	accuracy = total_correct * 100.0 / (double)total_observation;

	/* Check whether the current model achieved the best result. */
	if (ed->best_accuracy < accuracy) {
		ed->best_accuracy = accuracy;
		best = 1;
	}

	/* Report the accuracy. */
	fprintf(
		fpo,
		"%s: %d / %d (%3.2f%%)\n",
		(best == 1) ? "Best accuracy" : "Accuracy",
		total_correct,
		total_observation,
		accuracy
		);
	fflush(fpo);
#endif
#if 0
	fprintf(fpo, "Overall precision = %f (%d/%d)\n",
		(double)num_match / (double)num_extracted,
		num_match,
		num_extracted
		);
	fprintf(fpo, "Overall recall = %f (%d/%d)\n",
		(double)num_match / (double)num_to_extract,
		num_match,
		num_to_extract
		);
	fprintf(fpo, "Seconds = %f\n", clock() / (double)CLOCKS_PER_SEC);
#endif
	return 0;
}

int main_learn(int argc, char *argv[], const char *argv0)
{
	int i, ret = 0, arg_used = 0;
	learn_option_t opt;
	const char *command = argv[0];
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	crf_data_t data, eval;
	crf_trainer_t *trainer = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;
	evaluation_data_t ed;

	/* Parse the command-line option. */
	learn_option_init(&opt);
	arg_used = option_parse(++argv, --argc, parse_learn_options, &opt);
	if (arg_used < 0) {
		return -1;
	}

	/* Set a training file. */
	if (arg_used < argc) {
		opt.training = strdup(argv[arg_used]);
	} else {
		opt.training = strdup("-");	/* STDIN. */
	}

	/* Create dictionaries for attributes and labels. */
	ret = crf_create_instance("dictionary", (void**)&attrs);
	if (!ret) {
		ret = -1;
		fprintf(fpe, "ERROR: Failed to create a dictionary instance.\n");
		goto force_exit;
	}
	ret = crf_create_instance("dictionary", (void**)&labels);
	if (!ret) {
		ret = -1;
		fprintf(fpe, "ERROR: Failed to create a dictionary instance.\n");
		goto force_exit;
	}

	/* Create a trainer instance. */
	ret = crf_create_instance("trainer.crf1m", (void**)&trainer);
	if (!ret) {
		ret = -1;
		fprintf(fpe, "ERROR: Failed to create a trainer instance.\n");
		goto force_exit;
	}

	/* Set parameters. */
	for (i = 0;i < opt.num_params;++i) {
		char *value = NULL;
		char *name = opt.params[i];
		crf_params_t* params = trainer->params(trainer);
		
		/* Split the parameter argument by the first '=' character. */
		value = strchr(name, '=');
		if (value != NULL) {
			*value++ = 0;
		}

		params->set(params, name, value);
		params->release(params);
	}

	/* Initialize the training data and evaluation data. */
	crf_data_init(&data);
	crf_data_init(&eval);

	/*
		Read a training data.
	 */
	fp = (strcmp(opt.training, "-") == 0) ? fpi : fopen(opt.training, "r");
	if (fp == NULL) {
		ret = -1;
		fprintf(fpe, "ERROR: Failed to open the training data.\n");
		goto force_exit;		
	}

	/* Read the data. */
	fprintf(fpo, "Reading the training data\n");
	read_data(fp, fpo, &data, attrs, labels);
	if (fp != fpi) fclose(fp);

	/* Set parameters. */

	/* Report the statistics of the data. */
	fprintf(fpo, "Number of instances: %d\n", data.num_instances);
	fprintf(fpo, "Total number of items: %d\n", crf_data_totalitems(&data));
	fprintf(fpo, "Number of attributes: %d\n", labels->num(attrs));
	fprintf(fpo, "Number of labels: %d\n", labels->num(labels));
	fprintf(fpo, "\n");
	fflush(fpo);

	/*
		Read an evaluation data.
	 */
	if (opt.evaluation != NULL) {
		fp = fopen(opt.evaluation, "r");
		if (fp == NULL) {
			ret = -1;
			fprintf(fpe, "ERROR: Failed to open the evaluation data.\n");
			goto force_exit;		
		}
		fprintf(fpo, "Reading the evaluation data\n");
		read_data(fp, fpo, &eval, attrs, labels);
		fprintf(fpo, "Number of instances: %d\n", eval.num_instances);
		fprintf(fpo, "Number of total items: %d\n", crf_data_totalitems(&eval));
		fprintf(fpo, "\n");
		fflush(fpo);
		fclose(fp);
	}

	data.num_labels = labels->num(labels);
	data.num_attrs = labels->num(attrs);
	data.max_item_length = MAX(crf_data_maxlength(&data), crf_data_maxlength(&eval));

	ed.fpo = fpo;
	ed.best_accuracy = 0;
	crf_evaluation_init(&ed.eval, labels->num(labels));
	ed.attrs = attrs;
	ed.labels = labels;
	ed.out.num_labels = crf_data_maxlength(&eval);
	ed.out.labels = (int*)calloc(ed.out.num_labels, sizeof(int));
	ed.out.probability = 0;
	ed.data = &eval;

	trainer->set_message_callback(trainer, NULL, trainer_callback);
	trainer->set_evaluate_callback(trainer, &ed, evaluate_callback);
	trainer->trainer(trainer, &data);

	if (opt.model != NULL) {
		trainer->save(trainer, opt.model, attrs, labels);
	}

force_exit:
	crf_evaluation_finish(&ed.eval);
	learn_option_finish(&opt);
	return ret;
}
